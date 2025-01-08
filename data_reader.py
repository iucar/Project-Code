from frame import Frame
import csv

class DataReader:
    def __init__(self):
        """
        Class for reading data from CSV files and storing it in a Frame class format.
        Each row in CSV file is a different frame.
        """
        pass

    def load_data(self, file_name: str) -> list[Frame]:
        """
        Reads data from a CSV file and returns a list of Frame objects.
        """
        data = self._read_csv(file_name)
        frames = self._initialize_frames(len(data))
        self._populate_frames(data, frames)
        return frames
    
    def _read_csv(self, file_name: str) -> list[dict]:
        """Reads CSV data into a list of dictionaries."""
        try:
            with open(file_name, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                if not reader.fieldnames:
                    raise ValueError(f"CSV file {file_name} has no headers.")
                frames = list(reader)
                return frames
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_name}")
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_name}: {e}")

    def _initialize_frames(self, count: int) -> list[Frame]:
        """Initializes a list of Frame objects."""
        return [Frame() for _ in range(count)]

    def _populate_frames(self, data: list[dict], sequential_frames: list[Frame]) -> None:
        """Populates Frame objects with data."""
    
        for i in range(len(data)):

            sequential_frames[i].palm.position.x = float(data[i].get('palm_position_x', 0.0))
            sequential_frames[i].palm.position.y = float(data[i].get('palm_position_y', 0.0))
            sequential_frames[i].palm.position.z = float(data[i].get('palm_position_z', 0.0))

            sequential_frames[i].palm.normal.x = float(data[i].get('palm_normal_x', 0.0))
            sequential_frames[i].palm.normal.y = float(data[i].get('palm_normal_y', 0.0))
            sequential_frames[i].palm.normal.z = float(data[i].get('palm_normal_z', 0.0))

            sequential_frames[i].palm.direction.x = float(data[i].get('palm_direction_x', 0.0))
            sequential_frames[i].palm.direction.y = float(data[i].get('palm_direction_y', 0.0))
            sequential_frames[i].palm.direction.z = float(data[i].get('palm_direction_z', 0.0))

            sequential_frames[i].digits[1].bones[3].next_joint.x = float(data[i].get('digit1_bone3_next_joint_x', 0.0))
            sequential_frames[i].digits[1].bones[3].next_joint.y = float(data[i].get('digit1_bone3_next_joint_y', 0.0))
            sequential_frames[i].digits[1].bones[3].next_joint.z = float(data[i].get('digit1_bone3_next_joint_z', 0.0))
            

            sequential_frames[i].digits[2].bones[3].next_joint.x = float(data[i].get('digit2_bone3_next_joint_x', 0.0))
            sequential_frames[i].digits[2].bones[3].next_joint.y = float(data[i].get('digit2_bone3_next_joint_y', 0.0))
            sequential_frames[i].digits[2].bones[3].next_joint.z = float(data[i].get('digit2_bone3_next_joint_z', 0.0))

            sequential_frames[i].digits[3].bones[3].next_joint.x = float(data[i].get('digit3_bone3_next_joint_x', 0.0))
            sequential_frames[i].digits[3].bones[3].next_joint.y = float(data[i].get('digit3_bone3_next_joint_y', 0.0))
            sequential_frames[i].digits[3].bones[3].next_joint.z = float(data[i].get('digit3_bone3_next_joint_z', 0.0))

            sequential_frames[i].digits[4].bones[3].next_joint.x = float(data[i].get('digit4_bone3_next_joint_x', 0.0))
            sequential_frames[i].digits[4].bones[3].next_joint.y = float(data[i].get('digit4_bone3_next_joint_y', 0.0))
            sequential_frames[i].digits[4].bones[3].next_joint.z = float(data[i].get('digit4_bone3_next_joint_z', 0.0))
