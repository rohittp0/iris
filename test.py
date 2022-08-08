import glob


def main():
    files = glob.glob(r"data\IITD Database\**\*.*", recursive=True)
    with open("data/output1/README.md", "w") as readme:
        for i, file in enumerate(files):
            out = "/".join(file.replace(r"data\IITD Database", r"data\output1").split("\\"))
            out = f"![eye image](./{out}/eye.png)\n"

            print(out)
            readme.write(out)


if __name__ == "__main__":
    main()
