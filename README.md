# Transformer based model to OCR (for extracting table like data) from images and pdfs

This is a specific OCR that looks for Items and Costs columns in the image. 

This is composed of two models 

- Two state-of-the-art transformer based models to recognize tables and structure like columns.(microsoft/table-transformer-detection and microsoft/table-transformer-structure-recognition)
- Extract text using tesseract

Inputs : Either image or PDF
Outputs : {Table1 : [name: item1, amount: 50, cost_code: "" ]


How to run :
Google Colab : 
1. Upload requirements.txt to /content/sample_data/equirements.txt
2. !pip: -r "/content/sample_data/equirements.txt"
3. !sudo apt install poppler-utils tesseract-ocr
4. Copy app.py to a cell in the colab and run

Huggingface:
- url : https://huggingface.co/spaces/sussahoo/table_extraction





