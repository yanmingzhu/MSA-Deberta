# MSA-Deberta
Multimodal Sentiment Analysis with Integration of Deberta

## Download Data
Download data at https://drive.google.com/uc?id=1VJhSc2TGrPU8zJSVTYwn5kfuG47VaNQ3. Move the datafile to the data/ folder. The code expects the data file is data/mosei.pkl

## Training
python main.py --epochs=5 --batch_size=32 --fusion=[early|gated|late]

you can also run notebook msa_deberta.ipynb but it requires mounting the project folder on google drive.