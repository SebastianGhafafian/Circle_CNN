def main():
    """
    This function creates a data set as a .zip file
    The images are first created and are deleted again
    after the creation of the zip file    
    """
    import utils
    import os

    # Training data
    cwd = os.getcwd()
    # create n training images
    path = os.path.join(cwd, "data")
    utils.create_images(path,n=40000)
    # create a zip file of training data 
    zip_path = os.path.join(path, "dataset.zip")
    utils.make_archive(path, zip_path)
    # get all data in directory and delete .png and .csv data 
    dir = os.listdir(path)
    for item in dir:
        if item.endswith(".png") or item.endswith(".csv"):
            os.remove(os.path.join(path, item))


if __name__ == "__main__":
    main()