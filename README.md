## AI-Powered Lead Scoring Tool

## Overview
This project implements an AI-powered lead scoring tool designed to help sales teams prioritize high-quality leads. It uses a simple neural network trained on synthetic data with features derived from job titles, company size, and email validity. The model assigns a lead quality score between 0 and 1 to help optimize outreach efforts.
A user-friendly Streamlit web app allows users to upload CSV files of leads, deduplicates entries by email, scores each lead with the AI model, and lets users download the scored list for sales workflows.

## Features
•	AI lead quality scoring using a PyTorch MLP model
•	Input features:
  o	GloVe 50-dimensional embeddings of job titles
  o	Normalized company size
  o	Email validity check with regex
•	Deduplication of leads by email address
•	Simple Streamlit interface for uploading CSV and downloading scored leads
•	Lightweight and fast, suitable for real-time scoring

## Installation
1.	Clone the repository
2.	Install required packages:
pip install -r requirements.txt
3.	Download GloVe embeddings (50d) from https://nlp.stanford.edu/data/glove.6B.zip, unzip, and place the file glove.6B.50d.txt in the project folder.
4.	Place the pre-trained model file lead_scoring_model.pth in the project root folder.

## Usage
  •	Run the Streamlit web app
  •	Open the local URL in your browser, upload your leads CSV file, and get scored results with deduplication.
  •	CSV format should include columns:
  name, email, job_title, company, company_size, domain

## How It Works
1.	Job titles are converted into numeric vectors using pre-trained GloVe embeddings.
2.	Company size is normalized and combined with email validity features.
3.	The model predicts a lead quality score between 0 and 1.
4.	Leads are deduplicated based on email to avoid redundant contacts.
5.	Results are displayed and downloadable via the Streamlit app.

## Project Structure
•	indx.py: Streamlit frontend with lead scoring and deduplication logic
•	trainninference.ipynb: PyTorch lead scoring model and training code
•	glove.6B.50d.txt: GloVe embeddings file (not included, download separately)
•	lead_scoring_model.pth: Pre-trained PyTorch model weights
•	sample_leads.csv: Example leads CSV for testing
