import pathlib


class LineFixer:
    def fix(self, line):
        old_url = "https://raw.githubusercontent.com/cmower/spatial-casadi/master/doc/image/spatial-casadi.png"
        new_url = "spatial-casadi.png"
        line = line.replace(old_url, new_url)
        return line


def main():
    line_fixer = LineFixer()

    repo_path = pathlib.Path(__file__).parent.absolute().parent.absolute()
    doc_path = repo_path / "doc"
    readme_file_name = repo_path / "README.md"
    mainpage_file_name = doc_path / "mainpage.md"

    if mainpage_file_name.is_file():
        mainpage_file_name.unlink()
        print("Removed old version of doc/mainpage.md")

    with open(readme_file_name, "r") as input_file:
        with open(mainpage_file_name, "w") as output_file:
            for line in input_file.readlines():
                new_line = line_fixer.fix(line)
                output_file.write(new_line)

    print("Created doc/mainpage.md")


if __name__ == "__main__":
    main()
