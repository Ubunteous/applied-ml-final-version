from pathlib import Path
from dataclasses import dataclass

# known labels:
# labels = ["gender", "smiling", "face_shape", "eye_color"]
# data_dir = ["celeba", "cartoon_set"]

@dataclass
class B1():
    data_dir = "cartoon_set"
    label = "face_shape"
    max_data = 50
    proportion_train = 0.75
    task = "B1"
    
    def get_data_dir(self, debug = False):
        name = self.data_dir
        
        # the file is called from repository / Ax/ or Bx/
        cwd = Path(__file__).resolve().parents[1]

        # Options: celeba, cartoon_set and *_test
        # Each contains img/*.jpg and labels.csv
        basedir = cwd / "Datasets"
        # "celeba" or "cartoon_set" _test
        images_dir = basedir / name

        # print(f"Image dir: {images_dir}")

        if images_dir.exists() == False:
            print(f"Directory {images_dir.name} does not exist. Make sure your current directory is applied-ml-final-version")
            return None
    
        return images_dir


    def get_main_properties(self):
        return self.label, self.max_data, self.get_data_dir(), self.proportion_train
