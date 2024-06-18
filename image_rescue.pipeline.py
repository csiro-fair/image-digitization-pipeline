from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Tuple, Optional
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
from marimba.lib import image
from marimba.lib.decorators import multithreaded
from marimba.lib.parallel import multithreaded_generate_thumbnails


class ImageRescuePipeline(BasePipeline):
    """
    Marimba image rescue pipeline.
    """

    @staticmethod
    def get_pipeline_config_schema() -> dict:
        return {
            "platform_id": "PS1000",
        }

    @staticmethod
    def get_collection_config_schema() -> dict:
        return {
            "batch_id": "1a",
            "batch_data_path": "/datasets/work/oa-biaa-team/work/FAIR_for_imagery/WP8_DataRescue/FilmRescue/0823_FilmRescue_batch1a.csv",
            "inventory_data_path": "/datasets/work/oa-biaa-team/work/FAIR_for_imagery/WP8_DataRescue/FilmRescue/Film-Inventory_2023.xlsx",
            "import_path": "/datasets/work/oa-biaa-team/work/FAIR_for_imagery/WP8_DataRescue/FilmRescue/FilmRescue_batch1a/",
        }

    def _import(
        self,
        data_dir: Path,
        source_path: Path,
        config: Dict[str, Any],
        **kwargs: dict,
    ) -> None:
        self.logger.info(f"Importing data from {source_path} to {data_dir}")

        import_path = config.get("import_path")
        if import_path is None:
            self.logger.error("Config missing 'import_path'")
            return

        base_path = Path(import_path)
        if not source_path.is_dir():
            self.logger.error(f"Source path {source_path} is not a directory")
            return

        files_to_copy = [source_file for source_file in source_path.glob("**/*.jpg") if source_file.is_file()]

        @multithreaded(logger=self.logger)
        def copy_files(item: Path, data_dir: Path, base_path: Path) -> None:
            try:
                destination_path = data_dir / item.relative_to(base_path)
                destination_path.parent.mkdir(parents=True, exist_ok=True)

                if not self.dry_run:
                    copy2(item, destination_path)
                self.logger.debug(f"Copied {item.resolve().absolute()} -> {destination_path}")
            except Exception as e:
                self.logger.error(f"Failed to copy {item.resolve().absolute()}: {e}")

        # Call the decorated function
        copy_files(data_dir=data_dir, base_path=base_path, items=files_to_copy)

    def _process(
        self,
        data_dir: Path,
        config: Dict[str, Any],
        **kwargs: dict,
    ):
        # Copy CSV files to data directory and load into dataframes
        copy2(config.get("batch_data_path"), data_dir)
        batch_data_df = pd.read_csv(data_dir / Path(config.get("batch_data_path")).name)
        inventory_df = pd.read_excel(config.get("inventory_data_path"), sheet_name="SlideRescue_all")

        # Convert and save as CSV
        inventory_df.to_csv((data_dir / Path(config.get("inventory_data_path")).stem).with_suffix(".csv"), index=False)

        # Merge dataframes on common columns
        merged_df = pd.merge(inventory_df, batch_data_df, on="Survey_Stn")

        # Group the DataFrame by 'Survey_Stn' column
        grouped = merged_df.groupby("Survey_Stn")

        # Loop through each group
        for name, group in grouped:
            group_image_index = 1

            sorted_group = group.sort_values(by="sequence")
            print(sorted_group)

            # List to store all found jpg files
            group_jpg_files = []

            # Iterate through each directory and glob for jpg files
            for index, row in sorted_group.iterrows():
                camera_roll_path = data_dir / str(row["folder_name"]).zfill(8)

                # Skip if the directory does not exist
                if not camera_roll_path.exists():
                    print(f"Camera roll path does not exist: {camera_roll_path}")
                    continue

                folder_jpg_files = sorted(list(camera_roll_path.glob("*.jpg")), reverse=bool(row["order"]))
                folder_jpg_files_with_rotation = [(item, row["rotation "]) for item in folder_jpg_files]
                group_jpg_files.extend(folder_jpg_files_with_rotation)

            n_points = len(group_jpg_files)

            print(group_jpg_files)
            print(n_points)

            survey_id = group.iloc[0]["survey"]
            deployment_number = group.iloc[0]["deployment_no"]
            output_base_directory = (
                data_dir / survey_id / self._config.get("platform_id") / f"{survey_id}_{deployment_number}"
            )
            output_data_directory = output_base_directory / "data"
            output_stills_directory = output_base_directory / "stills"
            output_thumbnails_directory = output_base_directory / "thumbnails"

            # Collect start and end coordinates and timestamps
            start_lat, start_long = (
                group.iloc[0]["start_lat"],
                group.iloc[0]["start_long"],
            )
            end_lat, end_long = group.iloc[0]["end_lat"], group.iloc[0]["end_long"]
            start_time, end_time, utc_offset = (
                group.iloc[0]["Starte date/time"],
                group.iloc[0]["End Date/time"],
                group.iloc[0]["UTC offset"],
            )
            print(start_time, end_time)

            if not utc_offset:
                utc_offset = 10

            # Create a timedelta object for the offset
            offset = timedelta(hours=utc_offset)

            if start_time:
                # Adjust the timestamp for the UTC offset
                start_time = start_time + offset

            if end_time:
                # Adjust the timestamp for the UTC offset
                end_time = end_time + offset

            print(
                survey_id,
                deployment_number,
                output_stills_directory,
                start_lat,
                start_long,
                end_lat,
                end_long,
                start_time,
                end_time,
            )

            # Interpolate geo-coordinates and timestamps
            interpolated_points = self.interpolate_points(
                start_lat,
                start_long,
                end_lat,
                end_long,
                start_time,
                end_time,
                n_points,
            )
            print(interpolated_points)
            print(len(group_jpg_files), len(interpolated_points))

            # Prepare navigation file for the deployment
            column_mapping = {
                "VideoLabInventory": "videolab_inventory",
                "Proj": "project",
                "Gear": "platform_deployment",
                "Gear Component": "camera_name",
                "Area_name": "area_name",
                "transect_name": "transect_name",
                "NOTE": "notes",
                "Survey_PI": "survey_pi",
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
                "camera_name",
                "area_name",
                "transect_name",
                "approx_depth_range_in_metres",
                "notes",
                "survey_pi",
                "image_context",
                "abstract",
                "view_port",
            ]
            navigation_df = pd.DataFrame(columns=navigation_columns)

            # Process each JPG file
            for (jpg_file, rotation), (lat, long, time) in zip(group_jpg_files, interpolated_points):
                image_id = str(group_image_index).zfill(4)

                timestamp = time.strftime("%Y%m%dT%H%M%SZ")
                output_filename = (
                    f'{self._config.get("platform_id")}_{survey_id}_{deployment_number}_{timestamp}_{image_id}.JPG'
                )

                navigation_row = {
                    "filename": output_filename,
                    "platform_id": self._config.get("platform_id"),
                    "survey_id": survey_id,
                    "deployment_number": deployment_number,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "image_id": image_id,
                    "latitude": lat,
                    "longitude": long,
                }
                print(navigation_row)
                # Add data from column_mapping
                for col in column_mapping:
                    mapped_col = column_mapping.get(col)
                    if mapped_col and col in group.iloc[0]:
                        navigation_row[mapped_col] = group.iloc[0][col]
                else:
                    depth = group.iloc[0]["Depth approx range (m)"]
                    if depth:
                        depth_split = depth.split("-")
                        if len(depth_split) > 1:
                            navigation_row["approx_depth_range_in_metres"] = f"{min(depth_split)}-{min(depth_split)}"
                        else:
                            navigation_row["approx_depth_range_in_metres"] = f"{depth}-{depth}"

                print(navigation_row)
                navigation_df = navigation_df.append(navigation_row, ignore_index=True)

                input_file_path = camera_roll_path / jpg_file
                output_file_path = output_stills_directory / output_filename

                print(output_filename, rotation)
                if not output_file_path.exists():
                    output_stills_directory.mkdir(parents=True, exist_ok=True)
                    self.copy_and_rotate_image(input_file_path, output_file_path, rotation)
                    self.logger.debug(
                        f"Copied, sequenced, rotated and renamed {input_file_path.resolve().absolute()} -> {output_filename}"
                    )
                group_image_index += 1
            else:
                renamed_stills_list = list(output_stills_directory.glob("*.JPG"))

                if renamed_stills_list:
                    # Write out navigation data
                    navigation_data_path = (
                        output_data_directory / f'{self._config.get("platform_id")}_{survey_id}_{deployment_number}.CSV'
                    )
                    output_data_directory.mkdir(parents=True, exist_ok=True)
                    print(output_data_directory)
                    navigation_df.to_csv(navigation_data_path, index=False)

                    print("Generate thumbnails")
                    print(output_thumbnails_directory)

                    # Generate thumbnails using multithreading
                    thumbnail_list = multithreaded_generate_thumbnails(
                        self,
                        image_list=renamed_stills_list,
                        output_directory=output_base_directory / "thumbnails",
                    )

                    # Create an overview image from the generated thumbnails
                    thumbnail_overview_path = output_base_directory / "OVERVIEW.JPG"
                    image.create_grid_image(thumbnail_list, thumbnail_overview_path)

    def _package(
        self,
        data_dir: Path,
        config: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> Dict[Path, Tuple[Path, Optional[ImageData], Optional[Dict[str, Any]]]]:
        data_mapping: Dict[Path, Tuple[Path, Optional[List[ImageData]], Optional[Dict[str, Any]]]] = {}

        # List all files in the root directory recursively
        all_files = data_dir.glob("**/*")
        exclude_dir = data_dir / "stills"

        # Filter out files from the exclude directory
        ancillary_files = [f for f in all_files if exclude_dir not in f.parents]

        # Add ancillary files to data mapping
        for file_path in ancillary_files:
            if file_path.is_file():
                output_file_path = file_path.relative_to(data_dir)
                data_mapping[file_path] = output_file_path, None, None

        navigation_data_list = list(data_dir.glob("**/*.CSV"))

        for navigation_data in navigation_data_list:
            navigation_data_df = pd.read_csv(navigation_data)
            # navigation_data_df["timestamp"] = pd.to_datetime(navigation_data_df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f").dt.floor("S")

            # for file_path in file_paths:
            for index, row in navigation_data_df.iterrows():
                file_path = navigation_data.parent.parent / "stills" / row["filename"]
                output_file_path = file_path.relative_to(data_dir)

                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in [".jpg"]
                    and "_THUMB" not in file_path.name
                    and "overview" not in file_path.name
                ):
                    # TODO: This information should live in the collection.yml config then this can roll through that list
                    # Set the image creators
                    image_creators = [
                        ImagePI(name="Chris Jackett", orcid="0000-0003-1132-1558"),
                        ImagePI(name="Franzis Althaus", orcid="0000-0002-5336-4612"),
                        ImagePI(name="Candice Untiedt", orcid="0000-0003-1562-3473"),
                        ImagePI(name="David Webb", orcid="0000-0000-0000-0000"),
                        # ImagePI(name="Nic Bax", orcid="0000-0002-9697-4963"),
                        ImagePI(name="CSIRO", orcid=""),
                    ]

                    camera_housing_viewport = CameraHousingViewport(
                        viewport_type=row["view_port"],
                        viewport_optical_density=0.0,
                        viewport_thickness_millimeter=0.0,
                        viewport_extra_description=None,
                    )

                    image_pi = ImagePI(name=row["survey_pi"], orcid="")

                    # "image_id",
                    # "videolab_inventory",
                    # "platform_deployment",
                    # "area_name",
                    # "transect_name",
                    # "notes",

                    image_data_list = [
                        ImageData(
                            # iFDO core (required)
                            image_datetime=datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S.%f"),
                            image_latitude=float(row["latitude"]),
                            image_longitude=float(row["longitude"]),
                            # Note: Leave image_altitude (singular number) empty
                            image_altitude=None,
                            image_coordinate_reference_system="EPSG:4326",
                            image_coordinate_uncertainty_meters=None,
                            image_context=row["image_context"],
                            image_project=row["survey_id"],
                            image_event=f'{row["survey_id"]}_{row["deployment_number"]}',
                            image_platform=self.config.get("platform_id"),
                            image_sensor=str(row["camera_name"]).strip(),
                            image_uuid=str(uuid4()),
                            # Note: Marimba automatically calculates and injects the SHA256 hash during packaging
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
                            # TODO: Mention to Timm Schoening
                            # TODO: Also ask about mapping to EXIF
                            # image_scale_reference=ImageScaleReference.NONE,
                            image_illumination=ImageIllumination.ARTIFICIAL_LIGHT,
                            image_pixel_mag=ImagePixelMagnitude.CM,
                            image_marine_zone=ImageMarineZone.SEAFLOOR,
                            image_spectral_resolution=ImageSpectralResolution.RGB,
                            image_capture_mode=ImageCaptureMode.MANUAL,
                            image_fauna_attraction=ImageFaunaAttraction.NONE,
                            # image_area_square_meter: Optional[float] = None
                            # image_meters_above_ground: Optional[float] = None
                            # image_acquisition_settings: Optional[dict] = None
                            # image_camera_yaw_degrees: Optional[float] = None
                            # image_camera_pitch_degrees: Optional[float] = None
                            # image_camera_roll_degrees: Optional[float] = None
                            image_overlap_fraction=0,
                            image_datetime_format="%Y-%m-%d %H:%M:%S.%f",
                            # image_camera_pose: Optional[CameraPose] = None
                            image_camera_housing_viewport=camera_housing_viewport,
                            # image_flatport_parameters: Optional[FlatportParameters] = None
                            # image_domeport_parameters: Optional[DomeportParameters] = None
                            # image_camera_calibration_model: Optional[CameraCalibrationModel] = None
                            # image_photometric_calibration: Optional[PhotometricCalibration] = None
                            # image_objective: Optional[str] = None
                            image_target_environment="Benthic habitat",
                            # image_target_timescale: Optional[str] = None
                            # image_spatial_constraints: Optional[str] = None
                            # image_temporal_constraints: Optional[str] = None
                            # image_time_synchronization: Optional[str] = None
                            image_item_identification_scheme="<platform_id>_<survey_id>_<deployment_number>_<datetimestamp>_<image_id>.<ext>",
                            image_curation_protocol="Processed with Marimba v0.3",
                            #
                            # # iFDO content (optional)
                            image_entropy=0.0,
                            # image_particle_count: Optional[int] = None
                            image_average_color=[0, 0, 0],
                            # image_mpeg7_colorlayout: Optional[List[float]] = None
                            # image_mpeg7_colorstatistics: Optional[List[float]] = None
                            # image_mpeg7_colorstructure: Optional[List[float]] = None
                            # image_mpeg7_dominantcolor: Optional[List[float]] = None
                            # image_mpeg7_edgehistogram: Optional[List[float]] = None
                            # image_mpeg7_homogenoustexture: Optional[List[float]] = None
                            # image_mpeg7_stablecolor: Optional[List[float]] = None
                            # image_annotation_labels: Optional[List[ImageAnnotationLabel]] = None
                            # image_annotation_creators: Optional[List[ImageAnnotationCreator]] = None
                            # image_annotations: Optional[List[ImageAnnotation]] = None
                        )
                    ]

                    data_mapping[file_path] = output_file_path, image_data_list, row.to_dict()

        return data_mapping

    @staticmethod
    def copy_and_rotate_image(src_path, dest_path, rotation_flag):
        # Open the image with PIL
        with Image.open(src_path) as img:
            if rotation_flag == 1:
                img = img.rotate(180)

            exif_dict = piexif.load(img.info.get("exif", b""))
            exif_bytes = piexif.dump(exif_dict)
            img.save(dest_path, quality=100, exif=exif_bytes)

    @staticmethod
    # Function to interpolate geo-coordinates and timestamps
    def interpolate_points(start_lat, start_long, end_lat, end_long, start_time, end_time, n_points):
        lats, longs, times = [], [], []

        # Check for availability and data types before coordinate interpolation
        if all(pd.notna(val) for val in [start_lat, end_lat, start_long, end_long]):
            lats = np.linspace(float(start_lat), float(end_lat), n_points).tolist()
            longs = np.linspace(float(start_long), float(end_long), n_points).tolist()
        else:
            # If unavailable, use start_lat and start_long for all points if they are valid
            if (
                pd.notna(start_lat)
                and pd.notna(start_long)
                and isinstance(start_lat, (int, float))
                and isinstance(start_long, (int, float))
            ):
                lats = [start_lat] * n_points
                longs = [start_long] * n_points

        # Check for availability before timestamp interpolation
        if pd.notna(start_time) and pd.notna(end_time):
            times = pd.date_range(start_time, end_time, periods=n_points).tolist()
        else:
            # If end_time is unavailable, use start_time for all points if it is valid
            if pd.notna(start_time):
                times = [pd.Timestamp(start_time)] * n_points

        return list(zip(lats, longs, times))
