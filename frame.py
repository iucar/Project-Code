from enum import Enum

NUM_FINGERS = 5
NUM_BONES = 4

class Frame:
    """
    A frame represents the data captured by the vision sensor at an instant in time.
    It contains information about the position and orientation of the hand detected by the sensor.
    """
    
    def __init__(self):
        """Initialize a Frame with default components."""
        self.id = 0
        self.palm = Palm()
        self.arm = Arm()
        self.digits = [Digit() for _ in range(NUM_FINGERS)]   # where digit[0] is the thumb and digit[4] is the pinky
    
    def get_palm_position(self) -> list[float]:
        """Return the palm position as [x, y, z]."""
        return self._get_vector_as_list(self.palm.position)
    
    def get_palm_normal(self) -> list[float]:
        """Return the palm normal as [x, y, z]."""
        return self._get_vector_as_list(self.palm.normal)
    
    def get_palm_direction(self) -> list[float]:
        """Return the palm direction as [x, y, z]."""
        return self._get_vector_as_list(self.palm.direction)
    
    def get_palm_orientation(self) -> list[float]:
        """Return the palm orientation as [x, y, z, w]."""
        return self._get_quaternion_as_list(self.palm.orientation)
    
    def get_palm_velocity(self) -> list[float]:
        """Return the palm velocity as [x, y, z]."""
        return self._get_vector_as_list(self.palm.velocity)
    
    def get_arm_direction(self) -> list[float]:
        """Return the arm direction as [x, y, z, w]."""
        return self._get_quaternion_as_list(self.arm.rotation)
    
    def get_prev_joint_bone(self, finger_num, bone_num) -> list[float]:
        """
        Get the position of the previous joint of a specified bone on a finger as [x, y, z]
        where the previous joint is the one closer to the palm.
        """
        if not (0 <= finger_num < NUM_FINGERS):
            raise ValueError(f"Invalid finger number: {finger_num}. Must be between 0 and {NUM_FINGERS - 1}.")
        if not (0 <= bone_num < NUM_BONES):
            raise ValueError(f"Invalid bone number: {bone_num}. Must be between 0 and {NUM_BONES - 1}.")
        return self._get_vector_as_list(self.digits[finger_num].bones[bone_num].prev_joint)

    def get_next_joint_bone(self, finger_num, bone_num) -> list[float]:
        """
        Get the position of the next joint of a specified bone on a finger as [x, y, z]
        where the next joint is the one closer to the fingertip.
        """
        if not (0 <= finger_num < NUM_FINGERS):
            raise ValueError(f"Invalid finger number: {finger_num}. Must be between 0 and {NUM_FINGERS - 1}.")
        if not (0 <= bone_num < NUM_BONES):
            raise ValueError(f"Invalid bone number: {bone_num}. Must be between 0 and {NUM_BONES - 1}.")
        return self._get_vector_as_list(self.digits[finger_num].bones[bone_num].next_joint)
    
    def _get_vector_as_list(self, vector: "Vector") -> list[float]:
        """Convert a Vector to a list of [x, y, z]."""
        return [vector.x, vector.y, vector.z]

    def _get_quaternion_as_list(self, quaternion: "Quaternion") -> list[float]:
        """Convert a Quaternion to a list of [x, y, z, w]."""
        return [quaternion.x, quaternion.y, quaternion.z, quaternion.w]



class Palm:
    """Represents the palm of a hand."""
    def __init__(self):
        self.position = Vector()
        self.normal = Vector()
        self.direction = Vector()
        self.orientation = Quaternion()
        self.velocity = Vector()

class Arm:
    """Represents the arm containing joints and rotation."""
    def __init__(self):
        self.prev_joint = Vector()
        self.next_joint = Vector()
        self.rotation = Quaternion()

class Digit:
    """
    Represents a finger with its bones.
        METACARPAL = 0      # Bone closest to the palm
        PROXIMAL = 1        # Middle bone
        INTERMEDIATE = 2    # Bone closer to the fingertip
        DISTAL = 3          # Bone at the fingertip
    """
    def __init__(self):
        # Use a dictionary to associate BoneType with Bone objects
        self.bones = [Bone() for _ in range(NUM_BONES)] 

class Bone:
    """Represents a bone in a finger."""
    def __init__(self):
        self.width = 0.0
        self.prev_joint = Vector()
        self.next_joint = Vector()
        self.rotation = Quaternion()

class Vector:
    """Represents a 3D vector."""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

class Quaternion:
    """Represents a 3D rotation quaternion."""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
