import os
import io
import re
import base64
import tempfile
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, 'Systemic Tau Academic Report', 0, 0, 'R')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def convert_markdown_to_pdf(md_text: str) -> bytes:
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # regex for markdown images: ![alt](data:image/png;base64,XXXX)
    img_pattern = re.compile(r'!\[(.*?)\]\(data:image/[^;]+;base64,(.*?)\)')
    
    temp_dir = tempfile.mkdtemp()
    img_counter = 0
    
    lines = md_text.split('\n')
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            pdf.ln(4)
            continue
            
        # Check for images
        img_match = img_pattern.search(line_clean)
        if img_match:
            b64_data = img_match.group(2)
            try:
                img_data = base64.b64decode(b64_data)
                img_path = os.path.join(temp_dir, f"img_{img_counter}.png")
                with open(img_path, "wb") as f:
                    f.write(img_data)
                
                pdf.ln(2)
                # Auto scale image to page width minus margins
                pdf.image(img_path, x=15, w=180)
                pdf.ln(5)
                img_counter += 1
            except Exception as e:
                pdf.set_font("Arial", 'I', 10)
                pdf.set_text_color(200, 0, 0)
                pdf.cell(0, 6, f"[Image Render Error: {str(e)}]", ln=True)
            continue

        # Basic markdown stripping for PyFPDF which doesn't support inline styles
        # Strip ** and *
        text = line_clean.replace('**', '').replace('__', '')
        # Handle headers
        if text.startswith('# '):
            pdf.set_font("Arial", 'B', 16)
            pdf.set_text_color(0, 51, 102)
            pdf.ln(6)
            pdf.multi_cell(0, 8, text[2:].strip().encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(2)
        elif text.startswith('## '):
            pdf.set_font("Arial", 'B', 14)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(4)
            pdf.multi_cell(0, 7, text[3:].strip().encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(1)
        elif text.startswith('### '):
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(50, 50, 50)
            pdf.ln(2)
            pdf.multi_cell(0, 6, text[4:].strip().encode('latin-1', 'replace').decode('latin-1'))
        elif text.startswith('- ') or text.startswith('* '):
            pdf.set_font("Arial", '', 10)
            pdf.set_text_color(0, 0, 0)
            pdf.set_x(20)
            pdf.multi_cell(0, 5, "- " + text[2:].strip().encode('latin-1', 'replace').decode('latin-1'))
        elif text.startswith('> '):
            pdf.set_font("Arial", 'I', 10)
            pdf.set_text_color(100, 100, 100)
            pdf.set_x(20)
            pdf.multi_cell(0, 5, text[2:].strip().encode('latin-1', 'replace').decode('latin-1'))
        else:
            pdf.set_font("Arial", '', 10)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 5, text.encode('latin-1', 'replace').decode('latin-1'))

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return pdf_bytes
