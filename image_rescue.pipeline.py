"""Marimba Pipeline for the CSIRO Image Rescue project."""  # noqa: N999
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import copy2
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import piexif
from PIL import Image
from ifdo.models import (
    CameraHousingViewport,
    ImageAcquisition,
    ImageCaptureMode,
    ImageData,
    ImageDeployment,
    ImageFaunaAttraction,
    ImageIllumination,
    ImageMarineZone,
    ImageNavigation,
    ImagePI,
    ImagePixelMagnitude,
    ImageQuality,
    ImageSpectralResolution,
)

from marimba.core.pipeline import BasePipeline
from marimba.core.utils.constants import Operation
from marimba.core.wrappers.dataset import DatasetWrapper
from marimba.lib import image
from marimba.lib.concurrency import multithreaded_generate_image_thumbnails
from marimba.main import __version__


class ImageRescuePipeline(BasePipeline):
    """
    Marimba image rescue pipeline.
    """

    @staticmethod
    def get_pipeline_config_schema() -> dict[str, Any]:
        """
        Get the pipeline configuration schema for the Image Rescue Pipeline.

        Returns:
            dict: Configuration parameters for the pipeline
        """
        return {}

    @staticmethod
    def get_collection_config_schema() -> dict[str, Any]:
        """
        Get the collection configuration schema for the Image Rescue Pipeline.

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
        self.logger.info(f"Importing data from {source_path} to {data_dir}")

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
        operation_value = Operation(operation["_value_"]) if isinstance(operation, dict) else Operation(operation)

        try:
            for file in files_to_copy:
                destination_path = data_dir / file.relative_to(base_path)
                destination_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    if not self.dry_run:
                        if operation_value == Operation.move:
                            self.logger.debug(f"Moving file: {file} -> {destination_path}")
                            shutil.move(str(file), str(destination_path))
                        elif operation_value == Operation.copy:
                            self.logger.debug(f"Copying file: {file} -> {destination_path}")
                            shutil.copy2(str(file), str(destination_path))
                        elif operation_value == Operation.link:
                            self.logger.debug(f"Creating hard link: {file} -> {destination_path}")
                            os.link(str(file), str(destination_path))
                        else:
                            self.logger.error(f"Invalid operation type: {operation_value}")
                            self._handle_operation_error(operation_value)
                except Exception as e:
                    self.logger.exception(f"Failed to {operation_value} file {file} to {destination_path}: {e!s}")
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
            "image_context": pd.Series(dtype="str"),
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
                # Get the directory path for this group
                group_dir = data_dir / str(name)
                if group_dir.exists():
                    shutil.rmtree(group_dir)
                    self.logger.debug(f"Deleted directory for group: {name}")
                continue  # Skip processing this group

            self._process_survey_station(data_dir, group, processed_folders)

        # Clean up empty processed folders
        self._cleanup_empty_folders(processed_folders)

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
        self._process_images_and_navigation(
            group_jpg_files,
            output_info,
            geo_time_info,
            group,
        )

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
        if pd.isna(utc_offset):
            utc_offset = 10
        else:
            utc_offset = float(utc_offset)

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
            "image-context": "image_context",
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
            self.move_and_rotate_image(jpg_file, output_file_path, rotation)
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

        thumbnail_overview_path = output_info["base_dir"] / "overview.jpg"
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

    def _cleanup_empty_folders(self, processed_folders: set[Path]) -> None:
        """Remove empty folders after processing."""
        for folder in processed_folders:
            if not any(folder.iterdir()):
                folder.rmdir()
                self.logger.debug(f"Deleted empty folder: {folder}")

    @staticmethod
    def _generate_summaries(
        data_mapping: dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]],
        data_dir: Path,
    ) -> None:

        # Generate data summaries at the voyage and platform levels, and an iFDO at the deployment level
        summary_directories: set[str] = set()
        ifdo_directories: set[str] = set()

        # Constants for directory depth levels
        summary_dir_depth = 0
        summary_subdir_depth = 1
        ifdo_dir_depth = 2

        # Collect output directories
        for relative_dst, image_data_list, _ in data_mapping.values():
            if image_data_list:
                parts = relative_dst.parts
                if len(parts) > summary_dir_depth:
                    summary_directories.add(parts[0])
                if len(parts) > summary_subdir_depth:
                    summary_directories.add(str(Path(parts[0]) / parts[1]))
                if len(parts) > ifdo_dir_depth:
                    ifdo_directories.add(str(Path(parts[0]) / parts[1] / parts[2]))

        # Convert the sets to sorted lists
        summary_dirs: list[str] = sorted(summary_directories)
        ifdo_dirs: list[str] = sorted(ifdo_directories)

        # Subset the data_mapping to include only files in the summary directories
        for directory in summary_dirs:
            subset_data_mapping = {
                src.as_posix(): image_data_list
                for src, (relative_dst, image_data_list, _) in data_mapping.items()
                if str(relative_dst).startswith(directory) and image_data_list
            }

            # Create a dataset summary for each of these
            dataset_wrapper = DatasetWrapper(data_dir / directory, version=None, dry_run=True)
            dataset_wrapper.dry_run = False
            dataset_wrapper.summary_name = f"{Path(directory).name}.summary.md"
            dataset_wrapper.generate_dataset_summary(subset_data_mapping, progress=False)

            # Add the summary to the dataset mapping
            output_file_path = dataset_wrapper.summary_path.relative_to(data_dir)
            data_mapping[dataset_wrapper.summary_path] = output_file_path, None, None

        # Subset the data_mapping to include only files in the ifdo directories
        for directory in ifdo_dirs:
            subset_data_mapping = {
                relative_dst.relative_to(directory).as_posix(): image_data_list
                for src, (relative_dst, image_data_list, _) in data_mapping.items()
                if str(relative_dst).startswith(directory) and image_data_list
            }

            # Create a iFDO for each of these
            dataset_wrapper = DatasetWrapper(data_dir / directory, version=None, dry_run=True)
            dataset_wrapper.dry_run = False
            dataset_wrapper.metadata_name = f"{Path(directory).name}.ifdo.yml"
            dataset_wrapper.generate_ifdo(directory, subset_data_mapping, progress=False)

            # Add the iFDO to the dataset mapping
            output_file_path = dataset_wrapper.metadata_path.relative_to(data_dir)
            data_mapping[dataset_wrapper.metadata_path] = output_file_path, None, None

    def _package(
        self,
        data_dir: Path,
        config: dict[str, Any],  # noqa: ARG002
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> dict[Path, tuple[Path, ImageData | None, dict[str, Any] | None]]:
        data_mapping: dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]] = {}

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
                    and "overview" not in file_path.name
                ):
                    # Set the image creators
                    image_creators = [
                        ImagePI(name="Chris Jackett", orcid="0000-0003-1132-1558"),
                        ImagePI(name="Franzis Althaus", orcid="0000-0002-5336-4612"),
                        ImagePI(name="Candice Untiedt", orcid="0000-0003-1562-3473"),
                        ImagePI(name="David Webb", orcid="0000-0001-5847-7002"),
                        # ImagePI(name="Nic Bax", orcid="0000-0002-9697-4963"),
                    ]

                    camera_housing_viewport = CameraHousingViewport(
                        viewport_type=row["view_port"],
                        viewport_optical_density=0.0,
                        viewport_thickness_millimeter=0.0,
                        viewport_extra_description=None,
                    )

                    image_pi = ImagePI(name=row["survey_pi"], orcid=row["orcid"])

                    # "image_id",
                    # "videolab_inventory",
                    # "platform_deployment",
                    # "area_name",
                    # "transect_name",
                    # "notes",

                    # ruff: noqa: ERA001
                    image_data = ImageData(
                        # iFDO core
                        image_datetime=datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
                        .replace(tzinfo=timezone.utc),
                        image_latitude=float(row["latitude"]),
                        image_longitude=float(row["longitude"]),
                        # Note: Leave image_altitude (singular number) empty
                        image_altitude=None,
                        image_coordinate_reference_system="EPSG:4326",
                        image_coordinate_uncertainty_meters=None,
                        image_context=row["image_context"],
                        image_project=row["survey_id"],
                        image_event=f'{row["survey_id"]}_{row["deployment_number"]}',
                        image_platform=str(row["platform_name"]).strip(),
                        image_sensor="Slidefilm Camera",
                        image_uuid=str(uuid4()),
                        # image_hash_sha256=image_hash_sha256,
                        image_pi=image_pi,
                        image_creators=image_creators,
                        image_license="CC BY 4.0",
                        image_copyright="CSIRO",
                        image_abstract=row["abstract"],
                        #
                        # # iFDO capture (optional)
                        image_acquisition=ImageAcquisition.SLIDE,
                        image_quality=ImageQuality.PRODUCT,
                        image_deployment=ImageDeployment.SURVEY,
                        image_navigation=ImageNavigation.RECONSTRUCTED,
                        # image_scale_reference=ImageScaleReference.NONE,
                        image_illumination=ImageIllumination.ARTIFICIAL_LIGHT,
                        image_pixel_mag=ImagePixelMagnitude.CM,
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
                        # image_camera_pose: Optional[CameraPose] = None,
                        image_camera_housing_viewport=camera_housing_viewport,
                        # image_flatport_parameters: Optional[FlatportParameters] = None,
                        # image_domeport_parameters: Optional[DomeportParameters] = None,
                        # image_camera_calibration_model: Optional[CameraCalibrationModel] = None,
                        # image_photometric_calibration: Optional[PhotometricCalibration] = None,
                        # image_objective: Optional[str] = None
                        image_target_environment="Benthic habitat",
                        # image_target_timescale: Optional[str] = None,
                        # image_spatial_constraints: Optional[str] = None,
                        # image_temporal_constraints: Optional[str] = None,
                        # image_time_synchronization: Optional[str] = None,
                        image_item_identification_scheme="<platform_id>_<survey_id>_<deployment_number>_<datetimestamp>_<image_id>.<ext>",
                        image_curation_protocol=f"Slide-film scanned; "
                                                f"digitised images processed with Marimba v{__version__}",
                        #
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

                    data_mapping[file_path] = output_file_path, [image_data], row.to_dict()

        # Generate summaries and iFDOs
        self._generate_summaries(data_mapping, data_dir)

        return data_mapping

    @staticmethod
    def copy_and_rotate_image(
        src_path: str,
        dest_path: str,
        rotation_flag: int,
    ) -> None:
        """
        Copy and rotate an image.

        Args:
            src_path: Source path of the image file
            dest_path: Destination path for the processed image
            rotation_flag: Flag indicating rotation (1 for 180 degrees, 0 for no rotation)

        Returns:
            None
        """
        # Open the image with PIL
        with Image.open(src_path) as original_img:
            processed_img = original_img.rotate(180) if rotation_flag == 1 else original_img

            exif_dict = piexif.load(processed_img.info.get("exif", b""))
            exif_bytes = piexif.dump(exif_dict)
            processed_img.save(dest_path, quality=100, exif=exif_bytes)

    @staticmethod
    def move_and_rotate_image(
        src_path: Path,
        dest_path: Path,
        rotation_flag: int,
    ) -> None:
        """
        Move and rotate an image.

        Args:
            src_path: Source path of the image file
            dest_path: Destination path for the processed image
            rotation_flag: Flag indicating rotation (1 for 180 degrees, 0 for no rotation)

        Returns:
            None
        """
        # Open the image with PIL
        with Image.open(src_path) as original_img:
            # Create rotated image if needed, otherwise use original
            processed_img = original_img.rotate(180) if rotation_flag == 1 else original_img

            exif_dict = piexif.load(processed_img.info.get("exif", b""))
            exif_bytes = piexif.dump(exif_dict)
            processed_img.save(dest_path, quality=100, exif=exif_bytes)

        # Delete the original image
        Path(src_path).unlink()

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
