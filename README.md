# Vessel Type Image Classification - Capstone

Run order (Python 3.9+):
1. pip install -r requirements.txt
2. python 01_fetch_dataset.py      -> creates dataset/ (5 classes x 20 = 100 images) + attribution.csv
3. python 02_vessel_classifier.py  -> creates results/ (metrics.json, confusion_matrix.png, ...)
4. Copy the numbers and plots into Vessel_Classification_Report.docx and the PPTX (red text marks the blanks).
5. Zip the dataset/ folder for submission.
