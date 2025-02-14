"""Marimba Pipeline for the CSIRO Image Digitization project."""  # noqa: INP001
import os
import shutil
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import copy2
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import piexif
from ifdo.models import (
    ImageAcquisition,
    ImageCameraHousingViewport,
    ImageCaptureMode,
    ImageContext,
    ImageCreator,
    ImageData,
    ImageDeployment,
    ImageFaunaAttraction,
    ImageIllumination,
    ImageLicense,
    ImageMarineZone,
    ImageNavigation,
    ImagePI,
    ImagePixelMagnitude,
    ImageQuality,
    ImageSpectralResolution,
)
from PIL import Image

from marimba.core.pipeline import BasePipeline
from marimba.core.schemas.base import BaseMetadata
from marimba.core.schemas.ifdo import iFDOMetadata
from marimba.lib import image
from marimba.lib.concurrency import multithreaded_generate_image_thumbnails
from marimba.main import __version__


class ImageDigitizationPipeline(BasePipeline):
    """
    Marimba Pipeline for the CSIRO Image Digitization project.

    This class extends the BasePipeline to provide specific functionality for processing and digitizing image data from
    the CSIRO Image Digitization project. It handles importing, processing, and packaging of image files along with
    associated metadata and navigation information.

    Methods:
        __init__: Initialize the ImageDigitizationPipeline instance.
        get_pipeline_config_schema: Get the pipeline configuration schema.
        get_collection_config_schema: Get the collection configuration schema.
        _import: Import data from source to destination directory.
        create_navigation_df: Create an empty navigation DataFrame with proper dtypes.
        _process: Process the imported data, including image files and navigation information.
        _package: Package the processed data and generate necessary metadata files.
        copy_and_rotate_image: Copy and rotate an image file.
        interpolate_points: Interpolate geographic coordinates and timestamps between start and end points.
    """

    def __init__(
        self,
        root_path: str | Path,
        config: dict[str, Any] | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize a new Pipeline instance.

        Args:
            root_path (str | Path): Base directory path where the pipeline will store its data and configuration files.
            config (dict[str, Any] | None, optional): Pipeline configuration dictionary. If None, default configuration
             will be used. Defaults to None.
            dry_run (bool, optional): If True, prevents any filesystem modifications. Useful for validation and testing.
             Defaults to False.
        """
        super().__init__(
            root_path,
            config,
            dry_run=dry_run,
            metadata_class=iFDOMetadata,
        )

    @staticmethod
    def get_pipeline_config_schema() -> dict[str, Any]:
        """
        Get the pipeline configuration schema for the Image Digitization Pipeline.

        Returns:
            dict: Configuration parameters for the pipeline
        """
        return {}

    @staticmethod
    def get_collection_config_schema() -> dict[str, Any]:
        """
        Get the collection configuration schema for the Image Digitization Pipeline.

        Returns:
            dict: Configuration parameters for the collection
        """
        return {
            "batch_data_path": "",
            "inventory_data_path": "",
            "import_path": "",
        }

    def _import(
        self,
        data_dir: Path,
        source_path: Path,
        config: dict[str, Any],
        **kwargs: dict[str, Any],
    ) -> None:
        import_path = config.get("import_path")
        if import_path is None:
            self.logger.exception("Config missing 'import_path'")
            return

        base_path = Path(import_path)
        if not source_path.is_dir():
            self.logger.exception(f"Source path {source_path} is not a directory")
            return

        files_to_copy = [source_file for source_file in source_path.glob("**/*.jpg") if source_file.is_file()]
        operation = kwargs.get("operation", "link")

        try:
            for file in files_to_copy:
                destination_path = data_dir / file.relative_to(base_path)
                destination_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    if not self.dry_run:
                        if operation == "move":
                            self.logger.debug(f"Moving file: {file} -> {destination_path}")
                            shutil.move(str(file), str(destination_path))
                        elif operation == "copy":
                            self.logger.debug(f"Copying file: {file} -> {destination_path}")
                            shutil.copy2(str(file), str(destination_path))
                        elif operation == "link":
                            self.logger.debug(f"Creating hard link: {file} -> {destination_path}")
                            os.link(str(file), str(destination_path))
                        else:
                            self.logger.error(f"Invalid operation type: {operation}")
                except Exception as e:
                    self.logger.exception(f"Failed to {operation} file {file} to {destination_path}: {e!s}")
                    # Continue with next file instead of stopping entire process
                    continue

        except Exception as e:
            # Handle any unexpected errors that aren't related to individual file operations
            self.logger.exception(f"Unexpected error during import process: {e}")

    @staticmethod
    def create_navigation_df() -> pd.DataFrame:
        """Create an empty navigation DataFrame with proper dtypes."""
        return pd.DataFrame({
            "filename": pd.Series(dtype="str"),
            "platform_id": pd.Series(dtype="str"),
            "survey_id": pd.Series(dtype="str"),
            "deployment_number": pd.Series(dtype="str"),
            "timestamp": pd.Series(dtype="str"),
            "image_id": pd.Series(dtype="str"),
            "project": pd.Series(dtype="str"),
            "latitude": pd.Series(dtype="float64"),
            "longitude": pd.Series(dtype="float64"),
            "videolab_inventory": pd.Series(dtype="str"),
            "platform_deployment": pd.Series(dtype="str"),
            "platform_name": pd.Series(dtype="str"),
            "area_name": pd.Series(dtype="str"),
            "transect_name": pd.Series(dtype="str"),
            "approx_depth_range_in_metres": pd.Series(dtype="str"),
            "notes": pd.Series(dtype="str"),
            "survey_pi": pd.Series(dtype="str"),
            "orcid": pd.Series(dtype="str"),
            "image_context_name": pd.Series(dtype="str"),
            "image_context_uri": pd.Series(dtype="str"),
            "abstract": pd.Series(dtype="str"),
            "view_port": pd.Series(dtype="str"),
        })

    def _process(
        self,
        data_dir: Path,
        config: dict[str, Any],
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        # Load and prepare data
        batch_data_df, inventory_df = self._load_input_data(data_dir, config)
        merged_df = inventory_df.merge(batch_data_df, on="Survey_Stn")
        grouped = merged_df.groupby("Survey_Stn")

        # Track folders that have had images processed
        processed_folders: set[Path] = set()

        # Process each survey station group
        for name, group in grouped:
            self.logger.debug(f"Processing group: {name}")

            # Check if this group name contains SS199701
            if "SS199701" in str(name):
                # Delete each folder associated with this group
                for _index, row in group.iterrows():
                    folder_path = data_dir / str(row["folder_name"]).zfill(8)
                    if folder_path.exists():
                        shutil.rmtree(folder_path)
                        self.logger.debug(f"Deleted directory: {folder_path}")
                continue  # Skip processing this group

            self._process_survey_station(data_dir, group, processed_folders)

    def _load_input_data(self, data_dir: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare input data from batch and inventory files."""
        # Get paths from config with type checking
        batch_data_path = config.get("batch_data_path")
        if not batch_data_path:
            raise ValueError("batch_data_path not found in config")

        inventory_data_path = config.get("inventory_data_path")
        if not inventory_data_path:
            raise ValueError("inventory_data_path not found in config")

        # Copy and load batch data
        copy2(str(batch_data_path), data_dir)
        batch_data_df = pd.read_csv(data_dir / Path(batch_data_path).name)

        # Load and convert inventory data
        inventory_df = pd.read_excel(inventory_data_path, sheet_name="SlideRescue_all")
        inventory_df.to_csv(
            (data_dir / Path(inventory_data_path).stem).with_suffix(".csv"),
            index=False,
        )

        return batch_data_df, inventory_df

    def _process_survey_station(self, data_dir: Path, group: pd.DataFrame, processed_folders: set[Path]) -> None:
        """Process a single survey station group."""
        # Collect image files and their rotation information
        group_jpg_files = self._collect_image_files(data_dir, group, processed_folders)
        if not group_jpg_files:
            return

        # Get deployment information and prepare output directories
        output_info = self._prepare_output_directories(data_dir, group)

        # Get coordinates and timestamps
        geo_time_info = self._get_geo_time_info(group)

        self._log_processing_details(output_info, geo_time_info)

        # Process images and create navigation data
        try:
            self._process_images_and_navigation(
                group_jpg_files,
                output_info,
                geo_time_info,
                group,
            )

            # After successful processing, clean up source directories
            for jpg_file, _ in group_jpg_files:
                with suppress(FileNotFoundError):
                    jpg_file.unlink()  # Delete the source file

            # Clean up empty source directories
            for folder in processed_folders:
                if folder.exists() and not any(folder.iterdir()):
                    with suppress(OSError):
                        folder.rmdir()

        except Exception:
            self.logger.exception("Error processing survey station")
            raise

    def _collect_image_files(
        self,
        data_dir: Path,
        group: pd.DataFrame,
        processed_folders: set[Path],
    ) -> list[tuple[Path, int]]:
        """Collect all image files for a survey station."""
        group_jpg_files = []
        sorted_group = group.sort_values(by="sequence")

        for _index, row in sorted_group.iterrows():
            camera_roll_path = data_dir / str(row["folder_name"]).zfill(8)

            if not camera_roll_path.exists():
                self.logger.debug(f"Camera roll path does not exist: {camera_roll_path}")
                continue

            folder_jpg_files = sorted(camera_roll_path.glob("*.jpg"), reverse=bool(row["order"]))
            folder_jpg_files_with_rotation = [(item, row["rotation "]) for item in folder_jpg_files]
            group_jpg_files.extend(folder_jpg_files_with_rotation)

            if folder_jpg_files:
                processed_folders.add(camera_roll_path)

        return group_jpg_files

    def _prepare_output_directories(self, data_dir: Path, group: pd.DataFrame) -> dict[str, Any]:
        """Prepare output directory structure and return relevant paths."""
        survey_id = group.iloc[0]["survey"]
        deployment_number = group.iloc[0]["deployment_no"]
        platform_id = group.iloc[0]["Platform_abbreviation "]

        output_base_directory = data_dir / survey_id / platform_id / f"{survey_id}_{deployment_number}"
        return {
            "base_dir": output_base_directory,
            "data_dir": output_base_directory / "data",
            "stills_dir": output_base_directory / "stills",
            "thumbnails_dir": output_base_directory / "thumbnails",
            "survey_id": survey_id,
            "deployment_number": deployment_number,
            "platform_id": platform_id,
        }

    def _get_geo_time_info(self, group: pd.DataFrame) -> dict[str, Any]:
        """Extract and process geographic and temporal information."""
        start_lat, start_long = group.iloc[0]["start_lat"], group.iloc[0]["start_long"]
        end_lat, end_long = group.iloc[0]["end_lat"], group.iloc[0]["end_long"]
        start_time = group.iloc[0]["Starte date/time"]
        end_time = group.iloc[0]["End Date/time"]

        # Handle NaN value for UTC offset with a default value
        utc_offset = group.iloc[0]["UTC offset"]
        utc_offset = 10 if pd.isna(utc_offset) else float(utc_offset)

        # Adjust timestamps for UTC offset
        offset = timedelta(hours=utc_offset)
        if start_time:
            start_time += offset
        if end_time:
            end_time += offset

        return {
            "start_lat": start_lat,
            "start_long": start_long,
            "end_lat": end_lat,
            "end_long": end_long,
            "start_time": start_time,
            "end_time": end_time,
        }

    def _process_images_and_navigation(
        self,
        group_jpg_files: list[tuple[Path, int]],
        output_info: dict[str, Any],
        geo_time_info: dict[str, Any],
        group: pd.DataFrame,
    ) -> None:
        """Process images and create navigation data for a survey station."""
        n_points = len(group_jpg_files)
        interpolated_points = self.interpolate_points(
            geo_time_info["start_lat"],
            geo_time_info["start_long"],
            geo_time_info["end_lat"],
            geo_time_info["end_long"],
            geo_time_info["start_time"],
            geo_time_info["end_time"],
            n_points,
        )

        navigation_df = self._process_image_files(
            group_jpg_files,
            interpolated_points,
            output_info,
            group,
        )

        renamed_stills_list = list(output_info["stills_dir"].glob("*.JPG"))
        if renamed_stills_list:
            self._create_output_files(renamed_stills_list, navigation_df, output_info)

    def _process_image_files(
        self,
        group_jpg_files: list[tuple[Path, int]],
        interpolated_points: list[tuple[float | None, float | None, pd.Timestamp | None]],
        output_info: dict[str, Any],
        group: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process individual image files and create navigation data."""
        navigation_df = self.create_navigation_df()
        column_mapping = self._get_column_mapping()

        for group_image_index, ((jpg_file, rotation), (lat, long, time)) in enumerate(
            zip(group_jpg_files, interpolated_points, strict=False),
            start=1,
        ):
            image_id = str(group_image_index).zfill(4)
            timestamp = time.strftime("%Y%m%dT%H%M%SZ") if time is not None else None
            output_filename = (
                f"{output_info['platform_id']}_{output_info['survey_id']}_"
                f"{output_info['deployment_number']}_{timestamp}_{image_id}.JPG"
            )

            navigation_row = self._create_navigation_row(
                output_filename,
                output_info,
                image_id,
                group,
                column_mapping,
                time,
                lat,
                long,
            )
            navigation_df = pd.concat([navigation_df, pd.DataFrame([navigation_row])], ignore_index=True)

            self._process_single_image(
                jpg_file,
                output_filename,
                rotation,
                output_info["stills_dir"],
            )

        return navigation_df

    def _get_column_mapping(self) -> dict[str, str]:
        """Return the column mapping for navigation data."""
        return {
            "VideoLabInventory": "videolab_inventory",
            "Proj": "project",
            "Gear": "platform_deployment",
            "Gear Component": "platform_name",
            "Area_name": "area_name",
            "transect_name": "transect_name",
            "NOTE": "notes",
            "Survey_PI": "survey_pi",
            "orcid": "orcid",
            "Image-context-name": "image_context_name",
            "Image-context-uri": "image_context_uri",
            "Abstract": "abstract",
            "View_Port": "view_port",
        }

    def _create_navigation_row(
        self,
        output_filename: str,
        output_info: dict[str, Any],
        image_id: str,
        group: pd.DataFrame,
        column_mapping: dict[str, Any],
        time: datetime | None = None,
        lat: float | None = None,
        long: float | None = None,
    ) -> dict[str, Any]:
        """Create a navigation data row for an image."""
        navigation_row = {
            "filename": output_filename,
            "platform_id": output_info["platform_id"],
            "survey_id": output_info["survey_id"],
            "deployment_number": output_info["deployment_number"],
            "image_id": image_id,
        }

        # Add optional navigation data only if present
        if time is not None:
            navigation_row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S.%f")
        if lat is not None:
            navigation_row["latitude"] = lat
        if long is not None:
            navigation_row["longitude"] = long

        # Add mapped columns
        for col, mapped_col in column_mapping.items():
            if col in group.iloc[0]:
                navigation_row[mapped_col] = group.iloc[0][col]

        # Process depth information
        depth = group.iloc[0]["Depth approx range (m)"]
        if pd.notna(depth) and depth:
            if "-" in str(depth):
                depth_split = depth.split("-")
                navigation_row["approx_depth_range_in_metres"] = f"{min(depth_split)}-{max(depth_split)}"
            else:  # Single value
                navigation_row["approx_depth_range_in_metres"] = f"{depth}-{depth}"
        else:
            navigation_row["approx_depth_range_in_metres"] = None
        return navigation_row

    def _process_single_image(
        self,
        jpg_file: Path,
        output_filename: str,
        rotation: int,
        output_stills_dir: Path,
    ) -> None:
        """Process a single image file."""
        output_file_path = output_stills_dir / output_filename

        if not output_file_path.exists():
            output_stills_dir.mkdir(parents=True, exist_ok=True)
            self.copy_and_rotate_image(jpg_file, output_file_path, rotation)
            self.logger.debug(f"Processed image {jpg_file} -> {output_filename}")

    def _create_output_files(
        self,
        renamed_stills_list: list[Path],
        navigation_df: pd.DataFrame,
        output_info: dict[str, Any],
    ) -> None:
        """Create navigation data file and generate thumbnails."""
        # Save navigation data
        navigation_data_path = (
            output_info["data_dir"] /
            f"{output_info['platform_id']}_{output_info['survey_id']}_{output_info['deployment_number']}.CSV"
        )
        if not navigation_data_path.parent.exists():
            navigation_data_path.parent.mkdir(parents=True)
        if not navigation_data_path.exists():
            navigation_df.to_csv(navigation_data_path, index=False)
            self.logger.debug(f"Navigation data saved to {navigation_data_path}")

        # Generate thumbnails and overview image
        thumbnail_list = multithreaded_generate_image_thumbnails(
            self,
            image_list=renamed_stills_list,
            output_directory=output_info["thumbnails_dir"],
        )

        thumbnail_overview_path = (
            output_info["base_dir"] /
            f"{output_info['platform_id']}_"
            f"{output_info['survey_id']}_"
            f"{output_info['deployment_number']}_OVERVIEW.JPG"
        )
        image.create_grid_image(thumbnail_list, thumbnail_overview_path)
        self.logger.debug(f"Generated thumbnail overview image at {thumbnail_overview_path}")

    def _log_processing_details(self, output_info: dict[str, Any], geo_time_info: dict[str, Any]) -> None:
        """Log processing details for debugging."""
        self.logger.debug(
            f"Survey ID: {output_info['survey_id']}, "
            f"Deployment: {output_info['deployment_number']}, "
            f"Stills Dir: {output_info['stills_dir']}, "
            f"Start: ({geo_time_info['start_lat']}, {geo_time_info['start_long']}), "
            f"End: ({geo_time_info['end_lat']}, {geo_time_info['end_long']}), "
            f"Time: ({geo_time_info['start_time']}, {geo_time_info['end_time']})",
        )

    def _package(
        self,
        data_dir: Path,
        config: dict[str, Any],  # noqa: ARG002
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]]:

        # Initialise an empty dictionary to store file mappings
        data_mapping: dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]] = {}

        # List all files in the root directory recursively
        all_files = data_dir.glob("**/*")
        exclude_dir = data_dir / "stills"

        # Filter out files from the exclude directory
        ancillary_files = [f for f in all_files if exclude_dir not in f.parents]

        # Add ancillary files to data mapping
        for file_path in ancillary_files:
            if file_path.is_file() and file_path.suffix.lower() != ".csv#":
                output_file_path = file_path.relative_to(data_dir)
                data_mapping[file_path] = output_file_path, None, None

        navigation_data_list = list(data_dir.glob("**/*.CSV"))

        for navigation_data in navigation_data_list:
            navigation_data_df = pd.read_csv(navigation_data)

            for _index, row in navigation_data_df.iterrows():
                file_path = navigation_data.parent.parent / "stills" / row["filename"]
                output_file_path = file_path.relative_to(data_dir)

                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in [".jpg"]
                    and "_THUMB" not in file_path.name
                    and "_OVERVIEW" not in file_path.name
                ):
                    # Set the image PI and creators
                    image_pi = ImagePI(name=row["survey_pi"], uri=f"https://orcid.org/{row['orcid']}")
                    image_creators = [
                        ImageCreator(name=row["survey_pi"], uri=f"https://orcid.org/{row['orcid']}"),
                        ImageCreator(name="Candice Untiedt", uri="https://orcid.org/0000-0003-1562-3473"),
                        ImageCreator(name="Christopher Jackett", uri="https://orcid.org/0000-0003-1132-1558"),
                        ImageCreator(name="Franziska Althaus", uri="https://orcid.org/0000-0002-5336-4612"),
                        ImageCreator(name="David Webb", uri="https://orcid.org/0000-0001-5847-7002"),
                        ImageCreator(name="Ben Scoulding", uri="https://orcid.org/0000-0002-9358-736X"),
                        ImageCreator(name="CSIRO", uri="https://www.csiro.au"),
                    ]

                    camera_housing_viewport = ImageCameraHousingViewport(
                        viewport_type=row["view_port"],
                        viewport_optical_density=0.0,
                        viewport_thickness_millimeter=0.0,
                        viewport_extra_description=None,
                    )

                    # Create ImageContext and ImageLicense objects
                    image_context = ImageContext(name=str(row["image_context_name"]), uri=str(row["image_context_uri"]))
                    image_project = ImageContext(name=row["survey_id"])
                    image_event = ImageContext(name=output_file_path.stem)
                    image_platform = ImageContext(name=str(row["platform_name"]).strip())
                    image_sensor = ImageContext(name="Slidefilm Camera")
                    image_license = ImageLicense(
                        name="CC BY-NC 4.0",
                        uri="https://creativecommons.org/licenses/by-nc/4.0",
                    )

                    # ruff: noqa: ERA001
                    image_data = ImageData(
                        # iFDO core
                        image_datetime=datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
                        .replace(tzinfo=timezone.utc),
                        image_latitude=float(row["latitude"]) if pd.notna(row["latitude"]) else None,
                        image_longitude=float(row["longitude"]) if pd.notna(row["longitude"]) else None,
                        # Note: Leave image_altitude_meters empty
                        image_altitude_meters=None,
                        image_coordinate_reference_system="EPSG:4326",
                        image_coordinate_uncertainty_meters=None,
                        image_context=image_context,
                        image_project=image_project,
                        image_event=image_event,
                        image_platform=image_platform,
                        image_sensor=image_sensor,
                        image_uuid=str(uuid4()),
                        image_pi=image_pi,
                        image_creators=image_creators,
                        image_license=image_license,
                        image_copyright="CSIRO",
                        image_abstract=row["abstract"],

                        # # iFDO capture (optional)
                        image_acquisition=ImageAcquisition.SLIDE,
                        image_quality=ImageQuality.PRODUCT,
                        image_deployment=ImageDeployment.SURVEY,
                        image_navigation=ImageNavigation.RECONSTRUCTED,
                        # image_scale_reference=ImageScaleReference.NONE,
                        image_illumination=ImageIllumination.ARTIFICIAL_LIGHT,
                        image_pixel_magnitude=ImagePixelMagnitude.CM,
                        image_marine_zone=ImageMarineZone.SEAFLOOR,
                        image_spectral_resolution=ImageSpectralResolution.RGB,
                        image_capture_mode=ImageCaptureMode.MANUAL,
                        image_fauna_attraction=ImageFaunaAttraction.NONE,
                        # image_area_square_meter: Optional[float] = None,
                        # image_meters_above_ground: Optional[float] = None,
                        # image_acquisition_settings: Optional[dict] = None,
                        # image_camera_yaw_degrees: Optional[float] = None,
                        # image_camera_pitch_degrees: Optional[float] = None,
                        # image_camera_roll_degrees: Optional[float] = None,
                        image_overlap_fraction=0,
                        image_datetime_format="%Y-%m-%d %H:%M:%S.%f",
                        # image_camera_pose: Optional[ImageCameraPose] = None,
                        image_camera_housing_viewport=camera_housing_viewport,
                        # image_flatport_parameters: Optional[ImageFlatportParameters] = None,
                        # image_domeport_parameters: Optional[ImageDomeportParameters] = None,
                        # image_camera_calibration_model: Optional[ImageCameraCalibrationModel] = None,
                        # image_photometric_calibration: Optional[ImagePhotometricCalibration] = None,
                        # image_objective: Optional[str] = None
                        image_target_environment="Benthic habitat",
                        # image_target_timescale: Optional[str] = None,
                        # image_spatial_constraints: Optional[str] = None,
                        # image_temporal_constraints: Optional[str] = None,
                        # image_time_synchronization: Optional[str] = None,
                        image_item_identification_scheme="<platform_id>_<survey_id>_<deployment_number>_<datetimestamp>_<image_id>.<ext>",
                        image_curation_protocol=f"Slide-film scanned; "
                                                f"digitised images processed with Marimba v{__version__}",

                        # # iFDO content (optional)
                        # Note: Marimba automatically calculates injects image_entropy and image_average_color
                        # during packaging
                        # image_entropy=0.0,
                        # image_particle_count: Optional[int] = None,
                        # image_average_color=[0, 0, 0],
                        # image_mpeg7_colorlayout: Optional[List[float]] = None,
                        # image_mpeg7_colorstatistics: Optional[List[float]] = None,
                        # image_mpeg7_colorstructure: Optional[List[float]] = None,
                        # image_mpeg7_dominantcolor: Optional[List[float]] = None,
                        # image_mpeg7_edgehistogram: Optional[List[float]] = None,
                        # image_mpeg7_homogenoustexture: Optional[List[float]] = None,
                        # image_mpeg7_stablecolor: Optional[List[float]] = None,
                        # image_annotation_labels: Optional[List[ImageAnnotationLabel]] = None,
                        # image_annotation_creators: Optional[List[ImageAnnotationCreator]] = None,
                        # image_annotations: Optional[List[ImageAnnotation]] = None,
                    )

                    metadata = self._metadata_class(image_data)
                    data_mapping[file_path] = output_file_path, [metadata], row.to_dict()

        return data_mapping

    @staticmethod
    def copy_and_rotate_image(
        src_path: Path,
        dest_path: Path,
        rotation_flag: int,
    ) -> None:
        """
        Copy and rotate an image, preserving the original file.

        Args:
            src_path: Source path of the image file
            dest_path: Destination path for the processed image
            rotation_flag: Flag indicating rotation (1 for 180 degrees, 0 for no rotation)

        Returns:
            None
        """
        # First check if destination already exists
        if dest_path.exists():
            return

        try:
            # Open the image with PIL
            with Image.open(src_path) as original_img:
                # Create rotated image if needed, otherwise use original
                processed_img = original_img.rotate(180) if rotation_flag == 1 else original_img
                exif_dict = piexif.load(processed_img.info.get("exif", b""))
                exif_bytes = piexif.dump(exif_dict)
                processed_img.save(dest_path, quality=100, exif=exif_bytes)

        except FileNotFoundError as err:
            raise FileNotFoundError(f"Source image not found: {src_path}") from err
        except Exception:
            # If anything goes wrong during processing, ensure destination is cleaned up
            if dest_path.exists():
                dest_path.unlink()
            raise

    @staticmethod
    def interpolate_points(
        start_lat: float | None,
        start_long: float | None,
        end_lat: float | None,
        end_long: float | None,
        start_time: str | pd.Timestamp | None,
        end_time: str | pd.Timestamp | None,
        n_points: int,
    ) -> list[tuple[float | None, float | None, pd.Timestamp | None]]:
        """
        Interpolate geographic coordinates and timestamps between start and end points.

        Args:
            start_lat: Starting latitude
            start_long: Starting longitude
            end_lat: Ending latitude
            end_long: Ending longitude
            start_time: Starting timestamp
            end_time: Ending timestamp
            n_points: Number of points to interpolate

        Returns:
            List of tuples containing (latitude, longitude, timestamp)
        """
        lats, longs, times = [], [], []

        # Check for availability and data types before coordinate interpolation
        if (isinstance(start_lat, float) and isinstance(end_lat, float)
            and isinstance(start_long, float) and isinstance(end_long, float)):
            lats = np.linspace(start_lat, end_lat, n_points).tolist()
            longs = np.linspace(start_long, end_long, n_points).tolist()
        # If unavailable, use start_lat and start_long for all points if they are valid
        elif (
            pd.notna(start_lat)
            and pd.notna(start_long)
            and isinstance(start_lat, int | float)
            and isinstance(start_long, int | float)
        ):
            lats = [start_lat] * n_points
            longs = [start_long] * n_points

        # Check for availability before timestamp interpolation
        if pd.notna(start_time) and pd.notna(end_time):
            times = pd.date_range(start_time, end_time, periods=n_points).tolist()
        # If end_time is unavailable, use start_time for all points if it is valid
        elif pd.notna(start_time):
            times = [pd.Timestamp(start_time)] * n_points

        return list(zip(lats, longs, times, strict=False))
