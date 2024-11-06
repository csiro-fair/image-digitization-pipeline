"""Marimba Pipeline for the CSIRO Image Rescue project."""  # noqa: N999

from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import copy2
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import piexif
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
from PIL import Image

from marimba.core.pipeline import BasePipeline
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
            "batch_id": "1a",
            "batch_data_path": (
                "/datasets/work/oa-biaa-team/work/FAIR_for_imagery/"
                "WP8_DataRescue/FilmRescue/0823_FilmRescue_batch1a.csv"
            ),
            "inventory_data_path": (
                "/datasets/work/oa-biaa-team/work/FAIR_for_imagery/"
                "WP8_DataRescue/FilmRescue/Film-Inventory_2023.xlsx"
            ),
            "import_path": (
                "/datasets/work/oa-biaa-team/work/FAIR_for_imagery/"
                "WP8_DataRescue/FilmRescue/FilmRescue_batch1a/"
            ),
        }

    def _import(
            self,
            data_dir: Path,
            source_path: Path,
            config: dict[str, Any],
            **kwargs: dict,  # noqa: ARG002
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

        try:
            for file in files_to_copy:
                destination_path = data_dir / file.relative_to(base_path)
                destination_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    if not self.dry_run:
                        copy2(file, destination_path)
                    self.logger.debug(f"Copied {file.resolve().absolute()} -> {destination_path}")
                except OSError as e:
                    self.logger.exception(f"Failed to copy {file.resolve().absolute()}: {e}")
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
            **kwargs: dict,  # noqa: ARG002
    ) -> None:
        # Copy CSV files to data directory and load into dataframes
        copy2(config.get("batch_data_path"), data_dir)
        batch_data_df = pd.read_csv(data_dir / Path(config.get("batch_data_path")).name)
        inventory_df = pd.read_excel(config.get("inventory_data_path"), sheet_name="SlideRescue_all")

        # Convert and save as CSV
        inventory_df.to_csv((data_dir / Path(config.get("inventory_data_path")).stem).with_suffix(".csv"), index=False)

        # Merge dataframes on common columns
        merged_df = inventory_df.merge(batch_data_df, on="Survey_Stn")

        # Group the DataFrame by 'Survey_Stn' column
        grouped = merged_df.groupby("Survey_Stn")

        # Set to track folders that have images moved out of them
        processed_folders = set()

        # Loop through each group
        for name, group in grouped:
            group_image_index = 1
            sorted_group = group.sort_values(by="sequence")
            self.logger.debug(f"Processing group: {name}")

            # List to store all found jpg files
            group_jpg_files = []

            # Iterate through each directory and glob for jpg files
            for _index, row in sorted_group.iterrows():
                camera_roll_path = data_dir / str(row["folder_name"]).zfill(8)

                # Skip if the directory does not exist
                if not camera_roll_path.exists():
                    self.logger.debug(f"Camera roll path does not exist: {camera_roll_path}")
                    continue

                folder_jpg_files = sorted(camera_roll_path.glob("*.jpg"), reverse=bool(row["order"]))
                folder_jpg_files_with_rotation = [(item, row["rotation "]) for item in folder_jpg_files]
                group_jpg_files.extend(folder_jpg_files_with_rotation)

                # Track processed folders
                if folder_jpg_files:
                    processed_folders.add(camera_roll_path)

            n_points = len(group_jpg_files)

            survey_id = group.iloc[0]["survey"]
            deployment_number = group.iloc[0]["deployment_no"]
            platform_id = group.iloc[0]["Platform_abbreviation "]
            output_base_directory = data_dir / survey_id / platform_id / f"{survey_id}_{deployment_number}"
            output_data_directory = output_base_directory / "data"
            output_stills_directory = output_base_directory / "stills"
            output_thumbnails_directory = output_base_directory / "thumbnails"

            # Collect start and end coordinates and timestamps
            start_lat, start_long = group.iloc[0]["start_lat"], group.iloc[0]["start_long"]
            end_lat, end_long = group.iloc[0]["end_lat"], group.iloc[0]["end_long"]
            start_time, end_time, utc_offset = (
                group.iloc[0]["Starte date/time"],
                group.iloc[0]["End Date/time"],
                group.iloc[0]["UTC offset"],
            )

            if not utc_offset:
                utc_offset = 10

            # Create a timedelta object for the offset
            offset = timedelta(hours=utc_offset)

            # Adjust timestamps for the UTC offset
            if start_time:
                start_time = start_time + offset
            if end_time:
                end_time = end_time + offset

            self.logger.debug(
                f"Survey ID: {survey_id}, Deployment: {deployment_number}, Stills Dir: {output_stills_directory}, "
                f"Start: ({start_lat}, {start_long}), End: ({end_lat}, {end_long}), Time: ({start_time}, {end_time})",
            )

            # Interpolate geo-coordinates and timestamps
            interpolated_points = self.interpolate_points(
                start_lat, start_long, end_lat, end_long, start_time, end_time, n_points,
            )

            # Prepare navigation file for the deployment
            column_mapping = {
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

            navigation_columns = [
                "filename",
                "platform_id",
                "survey_id",
                "deployment_number",
                "timestamp",
                "image_id",
                "project",
                "latitude",
                "longitude",
                "videolab_inventory",
                "platform_deployment",
                "platform_name",
                "area_name",
                "transect_name",
                "approx_depth_range_in_metres",
                "notes",
                "survey_pi",
                "orcid",
                "image_context",
                "abstract",
                "view_port",
            ]
            navigation_df = pd.DataFrame(columns=navigation_columns)

            # Process each JPG file
            for (jpg_file, rotation), (lat, long, time) in zip(group_jpg_files, interpolated_points, strict=False):
                image_id = str(group_image_index).zfill(4)
                timestamp = time.strftime("%Y%m%dT%H%M%SZ")
                output_filename = f"{platform_id}_{survey_id}_{deployment_number}_{timestamp}_{image_id}.JPG"

                navigation_row = {
                    "filename": output_filename,
                    "platform_id": platform_id,
                    "survey_id": survey_id,
                    "deployment_number": deployment_number,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "image_id": image_id,
                    "latitude": lat,
                    "longitude": long,
                }

                # Add data from column_mapping
                for col in column_mapping:
                    mapped_col = column_mapping.get(col)
                    if mapped_col and col in group.iloc[0]:
                        navigation_row[mapped_col] = group.iloc[0][col]
                depth = group.iloc[0]["Depth approx range (m)"]
                if depth:
                    depth_split = depth.split("-")
                    if len(depth_split) > 1:
                        navigation_row["approx_depth_range_in_metres"] = f"{min(depth_split)}-{max(depth_split)}"
                    else:
                        navigation_row["approx_depth_range_in_metres"] = f"{depth}-{depth}"

                # Initialize empty navigation DataFrame with proper dtypes
                navigation_df = self.create_navigation_df()

                input_file_path = camera_roll_path / jpg_file
                output_file_path = output_stills_directory / output_filename

                if not output_file_path.exists():
                    output_stills_directory.mkdir(parents=True, exist_ok=True)
                    self.move_and_rotate_image(input_file_path, output_file_path, rotation)
                    self.logger.debug(f"Processed image {input_file_path} -> {output_filename}")

                group_image_index += 1

            renamed_stills_list = list(output_stills_directory.glob("*.JPG"))

            if renamed_stills_list:
                # Write out navigation data
                navigation_data_path = output_data_directory / f"{platform_id}_{survey_id}_{deployment_number}.CSV"
                output_data_directory.mkdir(parents=True, exist_ok=True)
                if not navigation_data_path.exists():
                    navigation_df.to_csv(navigation_data_path, index=False)
                    self.logger.debug(f"Navigation data saved to {navigation_data_path}")

                # Generate thumbnails
                thumbnail_list = multithreaded_generate_image_thumbnails(
                    self, image_list=renamed_stills_list, output_directory=output_thumbnails_directory,
                )

                # Create an overview image from the generated thumbnails
                thumbnail_overview_path = output_base_directory / "overview.jpg"
                image.create_grid_image(thumbnail_list, thumbnail_overview_path)
                self.logger.debug(f"Generated overview thumbnail at {thumbnail_overview_path}")

        # Check and delete empty folders
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
            src_path: str,
            dest_path: str,
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
        Path.unlink(src_path)

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
        if all(pd.notna(val) for val in [start_lat, end_lat, start_long, end_long]):
            lats = np.linspace(float(start_lat), float(end_lat), n_points).tolist()
            longs = np.linspace(float(start_long), float(end_long), n_points).tolist()
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
