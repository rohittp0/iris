import glob

from main import process_image


def main():
    files = glob.glob(r"data\CASIA-Iris-Twins\**\*.jpg", recursive=True)
    with open("README.md", "w") as readme:
        for i, file in enumerate(files):
            out = "/".join(file.replace(r"data\CASIA-Iris-Twins", r"data\output").split("\\"))
            out = f"![eye image](./{out}/eye.png)\n"

            print(out)
            readme.write(out)


if __name__ == "__main__":
    main()
