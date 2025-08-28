# Pneumonia X-ray Classification with PyTorch

This project demonstrates how deep learning can assist in **medical image analysis**, focusing on the classification of **chest X-ray scans** into *Normal* vs. *Pneumonia*.  
The goal is to explore transfer learning with PyTorch and apply modern computer vision techniques to a clinically relevant problem.  

---

## Motivation
Pneumonia is a common and serious lung infection, and chest X-rays are the primary tool for diagnosis.  
While radiologists are highly skilled, automated classification systems can support clinical decision-making, provide second opinions, and accelerate large-scale screening.  

This project is not meant as a medical product, but as a **learning exercise** in:
- Using PyTorch for computer vision,
- Applying transfer learning on real-world medical data,
- Building explainable AI models with Grad-CAM visualizations.

---

## Dataset

- **Source:** [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- ~5,800 labeled X-ray images
- Pre-split into **train / val / test** subsets
- Two classes:  
  - **NORMAL**  
  - **PNEUMONIA**