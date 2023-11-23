from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import piexif
from PIL import Image
from PIL.ExifTags import TAGS
from ifdo.models import ImageData
from marimba.core.pipeline import BasePipeline
from marimba.lib import image
from datetime import datetime, timedelta


class ImageRescuePipeline(BasePipeline):
    """
    Test pipeline. No-op.
    """

    @staticmethod
    def get_pipeline_config_schema() -> dict:
        return {}

    @staticmethod
    def get_collection_config_schema() -> dict:
        return {
            "batch_id": "1a",
        }

    def _import(
        self,
        data_dir: Path,
        source_paths: List[Path],
        config: Dict[str, Any],
        **kwargs: dict,
    ):
        self.logger.info(f"Importing data from {source_paths=} to {data_dir}")

        base_path = Path(
            "/datasets/work/oa-biaa-team/work/FAIR_for_imagery/WP8_DataRescue/FilmRescue/FilmRescue_batch1a/"
        )
        for source_path in source_paths:
            if not source_path.is_dir():
                continue

            for source_file in source_path.glob("**/*"):
                if source_file.is_file() and source_file.suffix.lower() == ".jpg":
                    destination_path = data_dir / source_file.relative_to(base_path)
                    destination_path.parent.mkdir(parents=True, exist_ok=True)

                    if not self.dry_run:
                        copy2(source_file, destination_path)
                    self.logger.debug(
                        f"Copied {source_file.resolve().absolute()} -> {data_dir}"
                    )

    def get_image_output_file_name(
        self, deployment_config: dict, file_path: Path, index: int
    ) -> str:
        try:
            image = Image.open(file_path)

            # Check if image has EXIF data
            if hasattr(image, "_getexif"):
                exif_data = image._getexif()
                if exif_data is not None:
                    # Loop through EXIF tags
                    for tag, value in exif_data.items():
                        tag_name = TAGS.get(tag, tag)
                        if tag_name == "DateTime":
                            date = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                            # Convert to ISO 8601 format
                            iso_timestamp = date.strftime("%Y%m%dT%H%M%SZ")

                            # Construct and return new filename
                            return (
                                f'{self.config.get("platform_id")}_'
                                f"SCP_"
                                f'{self.config.get("voyage_id").split("_")[0]}_'
                                f'{self.config.get("voyage_id").split("_")[1]}_'
                                f'{deployment_config.get("deployment_id").split("_")[2]}_'
                                f"{iso_timestamp}_"
                                f"{index:04d}"
                                f".JPG"
                            )
            else:
                self.logger.error(f"No EXIF DateTime tag found in image {file_path}")

        except IOError:
            self.logger.error(
                f"Error: Unable to open {file_path}. Are you sure it's an image?"
            )

    def _process(self, data_dir: Path, config: Dict[str, Any], **kwargs: dict):
        # Load CSV files into dataframes
        batch_data = "/datasets/work/oa-biaa-team/work/FAIR_for_imagery/WP8_DataRescue/FilmRescue/0823_FilmRescue_batch1a.csv"
        inventory = "/datasets/work/oa-biaa-team/work/FAIR_for_imagery/WP8_DataRescue/FilmRescue/Film-Inventory_2023.xlsx"

        copy2(batch_data, data_dir)

        # Read the copied Excel file
        batch_data_df = pd.read_csv(data_dir / "0823_FilmRescue_batch1a.csv")
        inventory_df = pd.read_excel(inventory, sheet_name="SEFHES")

        # Convert and save as CSV
        inventory_df.to_csv(
            (data_dir / Path(inventory).stem).with_suffix(".csv"), index=False
        )

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

                folder_jpg_files = sorted(
                    list(camera_roll_path.glob("*.jpg")), reverse=bool(not row["order"])
                )

                folder_jpg_files_with_rotation = [
                    (item, row["rotation "]) for item in folder_jpg_files
                ]

                group_jpg_files.extend(folder_jpg_files_with_rotation)

            n_points = len(group_jpg_files)

            print(group_jpg_files)
            print(n_points)

            survey_id = group.iloc[0]["survey"]
            deployment_no = group.iloc[0]["deployment_no"]
            output_directory = (
                data_dir
                / survey_id
                / "PS1000"
                / f"{survey_id}_{deployment_no}"
                / "stills"
            )

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
                deployment_no,
                output_directory,
                start_lat,
                start_long,
                end_lat,
                end_long,
                start_time,
                end_time,
            )

            # TODO: Generate nav-file for the deployment

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

            # Process each JPG file
            for (jpg_file, rotation), (lat, long, time) in zip(
                group_jpg_files, interpolated_points
            ):
                image_id = str(group_image_index).zfill(4)

                timestamp = time.strftime("%Y%m%dT%H%M%SZ")
                output_filename = f'{self.config.get("platform_id")}_{survey_id}_{deployment_no}_{timestamp}_{image_id}.JPG'

                input_file_path = camera_roll_path / jpg_file
                output_file_path = output_directory / output_filename

                # print(input_file_path, output_file_path, row["rotation"], lat, long, time)
                print(output_filename, rotation, lat, long, time)
                if not output_file_path.exists():
                    output_directory.mkdir(parents=True, exist_ok=True)
                    self.copy_and_rotate_image(
                        input_file_path,
                        output_file_path,
                        rotation,
                        gps_info=(lat, long),
                        time_info=time,
                    )
                    group_image_index += 1
            # exit()

            # TODO: Write out stats CSV - Candice
            # Directory structure... lat, longs

        # jpg_list = []
        #
        # for file in data_dir.glob("**/*"):
        #     if (
        #         file.is_file()
        #         and file.suffix.lower() in [".jpg"]
        #         and "_THUMB" not in file.name
        #         and "overview" not in file.name
        #     ):
        #         stills_path = data_dir / "stills"
        #         stills_path.mkdir(exist_ok=True)
        #
        #         # Rename images
        #         output_file_name = self.get_image_output_file_name(
        #             config, str(file), int(str(file).split("_")[-1].split(".")[0])
        #         )
        #         output_file_path = stills_path / output_file_name
        #         file.rename(output_file_path)
        #         self.logger.info(f"Renamed file {file.name} -> {output_file_path}")
        #
        #         jpg_list.append(output_file_path)
        #
        #     if file.is_file() and file.suffix.lower() in [".mp4"]:
        #         video_path = data_dir / "video"
        #         video_path.mkdir(exist_ok=True)
        #
        #         # Move videos
        #         output_file_path = video_path / file.name
        #         file.rename(output_file_path)
        #         self.logger.info(f"Renamed file {file.name} -> {output_file_path}")
        #
        #     if file.is_file() and file.suffix.lower() in [".csv"]:
        #         data_path = data_dir / "data"
        #         data_path.mkdir(exist_ok=True)
        #
        #         # Move data
        #         output_file_path = data_path / file.name
        #         file.rename(output_file_path)
        #         self.logger.info(f"Renamed file {file.name} -> {output_file_path}")
        #
        # thumb_list = []
        # thumbs_path = data_dir / "thumb"
        # thumbs_path.mkdir(exist_ok=True)
        #
        # for jpg in jpg_list:
        #     output_filename = jpg.stem + "_THUMB" + jpg.suffix
        #     output_path = thumbs_path / output_filename
        #     self.logger.info(f"Generating thumbnail image: {output_path}")
        #     image.resize_fit(jpg, 300, 300, output_path)
        #     thumb_list.append(output_path)
        #
        # thumbnail_overview_path = data_dir / "overview.jpg"
        # self.logger.info(
        #     f"Creating thumbnail overview image: {str(thumbnail_overview_path)}"
        # )
        # image.create_grid_image(thumb_list, data_dir / "overview.jpg")

    def _compose(
        self, data_dirs: List[Path], configs: List[Dict[str, Any]], **kwargs: dict
    ) -> Dict[Path, Tuple[Path, List[ImageData]]]:
        data_mapping = {}
        # for data_dir, config in zip(data_dirs, configs):
        #     file_paths = []
        #     file_paths.extend(data_dir.glob("**/*"))
        #     base_output_path = Path(config.get("deployment_id"))
        #
        #     sensor_data_df = pd.read_csv(next((data_dir / "data").glob("*.CSV")))
        #     sensor_data_df["FinalTime"] = pd.to_datetime(
        #         sensor_data_df["FinalTime"], format="%Y-%m-%d %H:%M:%S.%f"
        #     ).dt.floor("S")
        #
        #     for file_path in file_paths:
        #         output_file_path = base_output_path / file_path.relative_to(data_dir)
        #
        #         if (
        #             file_path.is_file()
        #             and file_path.suffix.lower() in [".jpg"]
        #             and "_THUMB" not in file_path.name
        #             and "overview" not in file_path.name
        #         ):
        #             iso_timestamp = file_path.name.split("_")[5]
        #             target_datetime = pd.to_datetime(
        #                 iso_timestamp, format="%Y%m%dT%H%M%SZ"
        #             )
        #             matching_rows = sensor_data_df[
        #                 sensor_data_df["FinalTime"] == target_datetime
        #             ]
        #
        #             if not matching_rows.empty:
        #                 # in iFDO, the image data list for an image is a list containing single ImageData
        #                 image_data_list = [
        #                     ImageData(
        #                         image_datetime=datetime.strptime(
        #                             iso_timestamp, "%Y%m%dT%H%M%SZ"
        #                         ),
        #                         image_latitude=matching_rows["UsblLatitude"].values[0],
        #                         image_longitude=float(
        #                             matching_rows["UsblLongitude"].values[0]
        #                         ),
        #                         image_depth=float(matching_rows["Altitude"].values[0]),
        #                         image_altitude=float(
        #                             matching_rows["Altitude"].values[0]
        #                         ),
        #                         image_event=str(matching_rows["Operation"].values[0]),
        #                         image_platform=self.config.get("platform_id"),
        #                         image_sensor=str(matching_rows["Camera"].values[0]),
        #                         image_camera_pitch_degrees=float(
        #                             matching_rows["Pitch"].values[0]
        #                         ),
        #                         image_camera_roll_degrees=float(
        #                             matching_rows["Roll"].values[0]
        #                         ),
        #                         image_uuid=str(uuid4()),
        #                         # image_pi=self.config.get("voyage_pi"),
        #                         image_creators=[],
        #                         image_license="MIT",
        #                         image_copyright="",
        #                         image_abstract=self.config.get("abstract"),
        #                     )
        #                 ]
        #
        #                 data_mapping[file_path] = output_file_path, image_data_list
        #
        #         elif file_path.is_file():
        #             data_mapping[file_path] = output_file_path, None

        return data_mapping

    def to_deg(self, value, loc):
        """Convert latitude and longitude to 3-element tuple in degrees as required by EXIF"""
        if value < 0:
            loc_value = loc[1]
        else:
            loc_value = loc[0]

        # TODO: Check why not decimal seconds
        abs_value = abs(value)
        deg = int(abs_value)
        min = int((abs_value - deg) * 60)
        sec = int((abs_value - deg - min / 60) * 3600)
        return (deg, 1), (min, 1), (sec, 1), loc_value

    def copy_and_rotate_image(
        self, src_path, dest_path, rotation_flag, gps_info=None, time_info=None
    ):
        # Open the image with PIL
        with Image.open(src_path) as img:
            if rotation_flag == 1:
                img = img.rotate(180)

            exif_dict = piexif.load(img.info.get("exif", b""))

            if gps_info:
                latitude, longitude = gps_info
                gps_ifd = {
                    piexif.GPSIFD.GPSLatitudeRef: self.to_deg(latitude, ["N", "S"])[-1],
                    piexif.GPSIFD.GPSLatitude: self.to_deg(latitude, ["N", "S"])[:-1],
                    piexif.GPSIFD.GPSLongitudeRef: self.to_deg(longitude, ["E", "W"])[
                        -1
                    ],
                    piexif.GPSIFD.GPSLongitude: self.to_deg(longitude, ["E", "W"])[:-1],
                }
                exif_dict["GPS"] = gps_ifd

            # TODO: Add custom VARS-compatible timestamp metadata atom
            if time_info:
                time_str = time_info.strftime("%Y:%m:%d %H:%M:%S").encode()
                exif_dict["0th"][piexif.ImageIFD.DateTime] = time_str
                exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = time_str
                # TODO: Think about DateTimeDigitized...
                # exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = time_str

            exif_bytes = piexif.dump(exif_dict)
            img.save(dest_path, exif=exif_bytes)

    # Function to interpolate geo-coordinates and timestamps
    def interpolate_points(
        self, start_lat, start_long, end_lat, end_long, start_time, end_time, n_points
    ):
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
