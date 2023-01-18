from pathlib import Path

def get_data_dir(name, debug = False):
    # utility to retrieve a dir containing datasets
    
    # the file is called from repository/tools/
    cwd = Path(__file__).resolve().parents[1]

    # Options: celeba, cartoon_set and *_test
    # Each contains img/*.jpg and labels.csv
    basedir = cwd / "Datasets"
    # "celeba" or "cartoon_set" _test
    images_dir = basedir / name

    if debug:
        print("Image dir:", images_dir)

    if images_dir.exists() == False:
        print(f"Directory {images_dir.name} does not exist. Make sure your current directory is applied-ml-final-version")
        return None
    
    return images_dir
