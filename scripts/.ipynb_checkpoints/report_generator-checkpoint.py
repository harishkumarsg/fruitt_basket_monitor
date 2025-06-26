# scripts/report_generator.py
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

def generate_report(results, annotated_img_path, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Title and timestamp
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Fruit Basket Rot Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 100
    for i, item in enumerate(results):
        line = f"{i+1}. Fruit: {item['name']}, Rot: {item['rot_percent']:.2f}%"
        c.drawString(50, y, line)
        y -= 20

    # Add annotated image if available
    if annotated_img_path and os.path.exists(annotated_img_path):
        try:
            c.drawImage(annotated_img_path, 50, 50, width=500, preserveAspectRatio=True, mask='auto')
        except:
            pass

    c.save()