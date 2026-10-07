# program to be used to extract scenes from the pdf ebooks
# unfortunately this is only working for entire books at the moment but adding indexes is not hard
# make sure to include the path to the ebook relative to where you are launching the program from
import pymupdf
from pathlib import Path

print("PROGRAM STARTED")

search = input("Exact filename including path: ")
path = Path(search)

if not path.exists():
    print("Error: File does not exist.")

elif not path.is_file():
    print("Error: Path is not a file.")

else:
    try:
        pdf = pymupdf.open(path)

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            pix = page.get_pixmap(dpi=200)
            pix.save(f"Waldo_{page_num + 1}.png")

        print("PDF successfully converted.")

    except pymupdf.FileDataError:
        print("Error: The file is not a valid PDF.")