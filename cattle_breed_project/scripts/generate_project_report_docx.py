#!/usr/bin/env python3
"""
Comprehensive Word Document Report Generator for Cattle & Buffalo Breed Classification System.
Creates an extensively detailed, publication-ready DOCX technical report containing
all dataset details, taxonomic tables, experimental benchmarks, architecture diagrams,
embedded evaluation charts, API specifications, and setup instructions.
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner padding for table cells."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)

def add_styled_heading(doc, text, level):
    """Add customized heading with consistent corporate theme styling."""
    heading = doc.add_heading(text, level=level)
    heading.paragraph_format.keep_with_next = True
    heading.paragraph_format.space_before = Pt(14)
    heading.paragraph_format.space_after = Pt(4)
    run = heading.runs[0] if heading.runs else heading.add_run(text)
    if level == 1:
        run.font.color.rgb = RGBColor(24, 76, 120)  # Deep Navy Blue
        run.font.size = Pt(18)
        run.font.bold = True
    elif level == 2:
        run.font.color.rgb = RGBColor(41, 128, 185)  # Medium Blue
        run.font.size = Pt(14)
        run.font.bold = True
    elif level == 3:
        run.font.color.rgb = RGBColor(52, 73, 94)  # Charcoal
        run.font.size = Pt(12)
        run.font.bold = True
    return heading

def add_callout(doc, text, title="NOTE / KEY FINDING"):
    """Add a highlighted callout box for important findings."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)
    
    cell = table.cell(0, 0)
    set_cell_background(cell, "F0F4F8")  # Very light blue/gray
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    # Left border styling in XML
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="1B4F72"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tc_pr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r_title = p.add_run(f"[{title}] ")
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(27, 79, 114)
    r_title.font.size = Pt(10)
    
    r_text = p.add_run(text)
    r_text.font.size = Pt(10)
    r_text.font.italic = True
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def format_table_headers(table, col_widths, headers, bg_hex="1B4F72"):
    """Format and style table header row."""
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], bg_hex)
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.size = Pt(9.5)
            
    # Set widths across all rows
    for row in table.rows:
        for i, width in enumerate(col_widths):
            row.cells[i].width = width

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    results_dir = os.path.join(project_dir, "results")
    output_docx_path = os.path.join(project_dir, "Cattle_Breed_Classification_Complete_Project_Report.docx")
    root_docx_path = os.path.abspath(os.path.join(project_dir, "..", "Cattle_Breed_Classification_Complete_Project_Report.docx"))

    print("Generating comprehensive technical report Word document...")
    doc = Document()

    # Set page margins (1 inch)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Base font styling
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(44, 62, 80)
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(6)

    # =========================================================================
    # TITLE & COVER BANNER
    # =========================================================================
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(24)
    title_p.paragraph_format.space_after = Pt(4)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run("AI-Powered Indigenous Cattle & Buffalo Breed Classification System")
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(24, 76, 120)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(18)
    run_sub = sub_p.add_run("Comprehensive Engineering Report, System Architecture & Experimental Technical Reference")
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(127, 140, 141)

    # Metadata Table
    meta_table = doc.add_table(rows=6, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_widths = [Inches(2.2), Inches(4.3)]
    meta_headers = ["Project Attribute", "Specification / Detail"]
    format_table_headers(meta_table, meta_widths, meta_headers, bg_hex="2C3E50")
    
    meta_data = [
        ("Target Classes", "17 Indigenous Bovine Breeds (8 Cattle + 9 Buffalo)"),
        ("Standard Reference", "ICAR - National Bureau of Animal Genetic Resources (NBAGR)"),
        ("Production Architecture", "ConvNeXt-Tiny (Pretrained ImageNet-1K, Fine-Tuned Custom Head)"),
        ("Core Tech Stack", "PyTorch 2.0+, Torchvision, Flask 3.0, React 18, Vite, TypeScript, Tailwind"),
        ("System Performance", "73.43% Test Accuracy | 0.7240 Macro F1 | ~40ms CPU Inference Latency")
    ]
    for row_idx, (k, v) in enumerate(meta_data, start=1):
        row_cells = meta_table.rows[row_idx].cells
        row_cells[0].text = k
        row_cells[1].text = v
        set_cell_background(row_cells[0], "F8F9F9")
        set_cell_background(row_cells[1], "FFFFFF")
        set_cell_margins(row_cells[0], top=80, bottom=80, left=100, right=100)
        set_cell_margins(row_cells[1], top=80, bottom=80, left=100, right=100)
        row_cells[0].paragraphs[0].runs[0].font.bold = True
        row_cells[0].paragraphs[0].runs[0].font.size = Pt(9.5)
        row_cells[1].paragraphs[0].runs[0].font.size = Pt(9.5)

    doc.add_page_break()

    # =========================================================================
    # 1. EXECUTIVE SUMMARY & ABSTRACT
    # =========================================================================
    add_styled_heading(doc, "1. Executive Summary & Abstract", level=1)
    
    doc.add_paragraph(
        "India possesses one of the richest and most diverse gene pools of indigenous bovine livestock in the world, "
        "comprising native cattle (Bos indicus) and riverine buffalo (Bubalus bubalis). These indigenous breeds exhibit "
        "phenotypic traits critical for climate resilience, including tropical disease resistance, heat tolerance, "
        "efficient roughage conversion, and high-fat A2 milk production. However, visual identification of indigenous breeds "
        "remains a critical bottleneck for veterinary field officers, artificial insemination technicians, dairy cooperatives, "
        "and smallholder farmers due to substantial morphological convergence between geographically proximate breeds."
    )
    doc.add_paragraph(
        "This project presents an end-to-end, production-ready Deep Learning Classification System designed to identify "
        "17 indigenous Indian breeds (8 cattle breeds and 9 buffalo breeds) cataloged under the National Bureau of Animal "
        "Genetic Resources (ICAR-NBAGR). Through rigorous dataset cleaning, duplicate elimination, perceptual hash clustering, "
        "and extensive architectural benchmarking across Convolutional and Transformer paradigms, the system establishes a "
        "reliable 73.43% test accuracy and 0.7240 Macro F1-score on an untouched, leak-free holdout test partition."
    )

    add_callout(
        doc,
        "The project eliminates legacy bugs including fake/mock prediction fallbacks, ResNet-50 checkpoint architecture "
        "mismatches, cross-breed contradictory image duplicates, and dataset leakage. It introduces an interactive "
        "React + Vite frontend with live HTML5 webcam scanning and an industrial Flask backend with singleton model initialization.",
        title="CORE MILESTONE ACHIEVED"
    )

    # =========================================================================
    # 2. SYSTEM ARCHITECTURE & COMPLETE TECH STACK
    # =========================================================================
    add_styled_heading(doc, "2. System Architecture & Complete Tech Stack", level=1)

    doc.add_paragraph(
        "The system operates as a decoupled, microservice-inspired full-stack machine learning application comprising "
        "four primary functional tiers: Data Engineering Tier, Model Inference Tier, RESTful API Gateway, and Interactive UI."
    )

    add_styled_heading(doc, "2.1 Technology Stack Specifications", level=2)

    tech_table = doc.add_table(rows=7, cols=3)
    tech_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tech_widths = [Inches(1.8), Inches(2.2), Inches(2.5)]
    format_table_headers(tech_table, tech_widths, ["Layer / Component", "Technology / Framework", "Purpose & Architectural Role"], bg_hex="1B4F72")

    tech_specs = [
        ("Deep Learning Framework", "PyTorch 2.0+ & Torchvision", "Model training, automatic differentiation, GPU/CPU tensor operations."),
        ("Vision Backbone", "ConvNeXt-Tiny & EfficientNet-B2", "Pretrained feature extractors with custom 17-class classifier heads."),
        ("Data Pipeline & Metrics", "Albumentations, PIL, Scikit-learn", "Realistic augmentations, perceptual dHash clustering, confusion matrices."),
        ("Backend Web API", "Flask 3.0 & Flask-CORS", "Singleton model serving, multipart image stream processing, JSON response routing."),
        ("Frontend Application", "React 18, Vite, TypeScript", "Single Page Application (SPA), state management, real-time webcam feed."),
        ("UI Component Library", "Tailwind CSS & Shadcn/UI (Radix)", "Accessible responsive card layouts, animated progress bars, badges.")
    ]
    for row_idx, (c1, c2, c3) in enumerate(tech_specs, start=1):
        r_cells = tech_table.rows[row_idx].cells
        r_cells[0].text = c1
        r_cells[1].text = c2
        r_cells[2].text = c3
        bg = "F8F9F9" if row_idx % 2 == 1 else "FFFFFF"
        for i in range(3):
            set_cell_background(r_cells[i], bg)
            set_cell_margins(r_cells[i], top=60, bottom=60, left=80, right=80)
            r_cells[i].paragraphs[0].runs[0].font.size = Pt(9)
        r_cells[0].paragraphs[0].runs[0].font.bold = True

    # =========================================================================
    # 3. VERIFIED 17-BREED TAXONOMY & CHARACTERISTICS
    # =========================================================================
    doc.add_page_break()
    add_styled_heading(doc, "3. ICAR-NBAGR Verified Breed Taxonomy & Descriptors", level=1)

    doc.add_paragraph(
        "All 17 breeds supported by this classification system are officially recognized by the Indian Council of Agricultural "
        "Research - National Bureau of Animal Genetic Resources (ICAR-NBAGR). Below is the comprehensive taxonomic registry:"
    )

    breed_table = doc.add_table(rows=18, cols=5)
    breed_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    breed_widths = [Inches(1.1), Inches(0.8), Inches(1.4), Inches(1.4), Inches(1.8)]
    format_table_headers(breed_table, breed_widths, ["Breed", "Species", "State of Origin", "Accession No.", "Key Physical Hallmarks"], bg_hex="273746")

    breed_records = [
        ("Amritmahal", "Cattle", "Karnataka", "0800_AMRITMAHAL_03001", "Iron-grey coat; long sharp upward horns; compact draught build."),
        ("Gir", "Cattle", "Gujarat", "0400_GIR_03007", "Convex domed forehead; pendulous folded ears; mottled red coat."),
        ("Hallikar", "Cattle", "Karnataka", "0800_HALLIKAR_03008", "Dark grey coat; long erect backward horns curving inward; agile trotting."),
        ("Kankrej", "Cattle", "Gujarat", "0417_KANKREJ_03011", "Lyre-shaped horns; silver-grey color; majestic high-stepping gait."),
        ("Krishna Valley", "Cattle", "Karnataka/MH", "0800_KRISHNAVALLEY_03015", "Massive heavy body; deep chest; grey-white with dark extremities."),
        ("Nimari", "Cattle", "Madhya Pradesh", "1000_NIMARI_03019", "Copper-red coat with white splashes; moderately convex forehead."),
        ("Ongole", "Cattle", "Andhra Pradesh", "0100_ONGOLE_03020", "Glossy white coat; majestic muscular hump; large pendulous dewlap."),
        ("Vechur", "Cattle", "Kerala", "0900_VECHUR_03030", "World's smallest cattle (<90cm); tiny forward horns; medicinal A2 milk."),
        ("Banni", "Buffalo", "Gujarat", "0400_BANNI_01011", "Solid black/copper coat; tightly coiled vertical horns; nocturnal grazing."),
        ("Bargur", "Buffalo", "Tamil Nadu", "1800_BARGUR_01015", "Brown-black coat; semicircular horns pointing backward/upward; agile."),
        ("Bhadawari", "Buffalo", "Uttar Pradesh", "2010_BHADAWARI_01003", "Copper/straw-colored coat; two white neck chevrons; milk fat up to 13%."),
        ("Jaffarabadi", "Buffalo", "Gujarat", "0400_JAFFARABADI_01006", "Largest river buffalo; bulging forehead; heavy drooping curved horns."),
        ("Mehsana", "Buffalo", "Gujarat", "0400_MEHSANA_01004", "Murrah-Surti cross; sickle-curved horns; persistent lactation cycle."),
        ("Murrah", "Buffalo", "Haryana", "0500_MURRAH_01001", "Jet black coat; tightly coiled spiral horns; premier dairy conformation."),
        ("Nagpuri", "Buffalo", "Maharashtra", "1100_NAGPURI_01007", "Long flat sword-shaped horns sweeping to shoulders; heat tolerant."),
        ("Nili Ravi", "Buffalo", "Punjab", "1600_NILIRAVI_01002", "'Panj Kalyan' (5 white patches: forehead, muzzle, 4 legs, tail); walled eyes."),
        ("Toda", "Buffalo", "Tamil Nadu", "0018_TODA_01010", "Wide crescent horns curving inward; thick fawn coat; Nilgiris endemic.")
    ]

    for row_idx, data in enumerate(breed_records, start=1):
        r_cells = breed_table.rows[row_idx].cells
        for col_idx in range(5):
            r_cells[col_idx].text = data[col_idx]
            bg = "EBF5FB" if data[1] == "Cattle" else "FEF9E7"
            set_cell_background(r_cells[col_idx], bg)
            set_cell_margins(r_cells[col_idx], top=50, bottom=50, left=70, right=70)
            r_cells[col_idx].paragraphs[0].runs[0].font.size = Pt(8.5)
        r_cells[0].paragraphs[0].runs[0].font.bold = True

    # =========================================================================
    # 4. DATASET ENGINEERING & LEAK-FREE SPLITTING
    # =========================================================================
    add_styled_heading(doc, "4. Dataset Engineering & Leak-Free Splitting", level=1)

    doc.add_paragraph(
        "A critical flaw identified during the technical audit of the original repository was data leakage caused by "
        "near-identical internet scraped images appearing across training and test subsets. To establish scientific integrity, "
        "a rigorous multi-stage dataset preparation pipeline was implemented:"
    )

    doc.add_paragraph(
        "1. Exact Deduplication (MD5 Hashing): Identified and purged 82 cross-breed contradictory duplicate images "
        "and 25 within-breed redundant images, preserving the raw dataset completely untouched in data/raw.\n"
        "2. Format Standardization: Converted all image streams into standardized 3-channel 8-bit RGB JPEGs, "
        "eliminating corrupt headers, truncated streams, and palette paletted formats.\n"
        "3. Perceptual Hash Clustering (dHash): Generated 64-bit difference hashes for all 1,820 images. Pairs exhibiting "
        "a Hamming distance <= 3 were clustered via Disjoint-Set Union (Union-Find) into 1,794 atomic clusters.\n"
        "4. Cluster-Aware Stratified Splitting: Partitioned entire clusters as atomic units into 70% Train, 15% Validation, "
        "and 15% Holdout Test splits. This guaranteed zero train-test data leakage."
    )

    # Split Distribution Table
    split_table = doc.add_table(rows=4, cols=5)
    split_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    split_widths = [Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.4)]
    format_table_headers(split_table, split_widths, ["Partition Name", "Total Images", "Cattle Images", "Buffalo Images", "Cross-Split Leakage"], bg_hex="1B4F72")

    split_rows = [
        ("Training Set (train.csv)", "1,278 (70.2%)", "598", "680", "0 overlaps"),
        ("Validation Set (val.csv)", "271 (14.9%)", "127", "144", "0 overlaps"),
        ("Holdout Test Set (test.csv)", "271 (14.9%)", "127", "144", "0 overlaps (Untouched)")
    ]
    for row_idx, r in enumerate(split_rows, start=1):
        r_cells = split_table.rows[row_idx].cells
        for col_idx in range(5):
            r_cells[col_idx].text = r[col_idx]
            set_cell_background(r_cells[col_idx], "FFFFFF" if row_idx % 2 == 0 else "F8F9F9")
            set_cell_margins(r_cells[col_idx], top=60, bottom=60, left=80, right=80)
            r_cells[col_idx].paragraphs[0].runs[0].font.size = Pt(9)
        r_cells[0].paragraphs[0].runs[0].font.bold = True
        r_cells[4].paragraphs[0].runs[0].font.bold = True

    # =========================================================================
    # 5. MODEL ARCHITECTURE BENCHMARKING & EXPERIMENTAL RESULTS
    # =========================================================================
    doc.add_page_break()
    add_styled_heading(doc, "5. Architectural Benchmarking & Experimental Results", level=1)

    doc.add_paragraph(
        "To objectively determine the best architecture, four distinct paradigms were trained and evaluated "
        "using identical dataset splits, class-weighted cross-entropy loss with label smoothing (alpha=0.1), and AdamW optimization:"
    )

    # Architecture Benchmark Table
    arch_table = doc.add_table(rows=5, cols=6)
    arch_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    arch_widths = [Inches(1.5), Inches(1.1), Inches(0.9), Inches(1.0), Inches(1.0), Inches(1.0)]
    format_table_headers(arch_table, arch_widths, ["Architecture", "Strategy", "Epochs", "Val Accuracy", "Val Macro F1", "Outcome"], bg_hex="16A085")

    arch_data = [
        ("ConvNeXt-Tiny", "Direct 17-Class", "10", "70.11%", "0.6788", "SELECTED BEST"),
        ("Swin-T (Transformer)", "Direct 17-Class", "10", "69.37%", "0.6732", "Competitive"),
        ("EfficientNet-B2", "Direct 17-Class", "15", "64.21%", "0.6233", "Baseline"),
        ("Hierarchical Pipeline", "Species -> Specialist", "21", "57.93%", "0.5479", "Rejected")
    ]
    for row_idx, r in enumerate(arch_data, start=1):
        r_cells = arch_table.rows[row_idx].cells
        for col_idx in range(6):
            r_cells[col_idx].text = r[col_idx]
            bg = "D4EFDF" if "SELECTED" in r[5] else ("FADBD8" if "Rejected" in r[5] else "FFFFFF")
            set_cell_background(r_cells[col_idx], bg)
            set_cell_margins(r_cells[col_idx], top=60, bottom=60, left=80, right=80)
            r_cells[col_idx].paragraphs[0].runs[0].font.size = Pt(9)
        r_cells[0].paragraphs[0].runs[0].font.bold = True
        r_cells[5].paragraphs[0].runs[0].font.bold = True

    add_styled_heading(doc, "5.1 Analysis: Why ConvNeXt-Tiny Won & Hierarchical Failed", level=2)
    doc.add_paragraph(
        "ConvNeXt-Tiny achieved the highest validation performance (70.11% accuracy, 0.6788 Macro F1) "
        "due to its 7x7 depthwise convolutions and large effective receptive field. This architecture effectively captured "
        "both global silhouette geometry (such as horn span and body posture) and fine local features (such as facial skin folds and dewlaps)."
    )
    doc.add_paragraph(
        "Conversely, Hierarchical Classification underperformed significantly (57.93% accuracy) due to Cascading Error Propagation: "
        "Although Stage 1 (Species Classifier) achieved 95.57% accuracy, the 4.43% misclassified samples were routed to the wrong specialist branch, "
        "guaranteeing a 100% error rate for those samples. Furthermore, partitioning the training dataset into Cattle (598 images) and "
        "Buffalo (680 images) starved the specialist backbones of visual diversity."
    )

    # Embed Training Curves Plot
    curves_img = os.path.join(results_dir, "training_validation_curves.png")
    if os.path.exists(curves_img):
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        doc.add_picture(curves_img, width=Inches(6.2))
        p_cap = doc.add_paragraph("Figure 1: ConvNeXt-Tiny Training & Validation Convergence Curves (Loss, Accuracy, and Macro F1).")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5)
        p_cap.runs[0].font.italic = True

    # =========================================================================
    # 6. FINAL UNTOUCHED TEST SET EVALUATION
    # =========================================================================
    doc.add_page_break()
    add_styled_heading(doc, "6. Final Evaluation on Untouched Test Dataset", level=1)

    doc.add_paragraph(
        "Following model selection, ConvNeXt-Tiny and EfficientNet-B2 were evaluated on the strictly isolated test dataset "
        "(annotations/test.csv, 271 samples). Neither test images nor labels were observed during training or hyperparameter tuning:"
    )

    # Test Results Comparison Table
    test_comp_table = doc.add_table(rows=6, cols=4)
    test_comp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    test_widths = [Inches(2.2), Inches(1.4), Inches(1.4), Inches(1.5)]
    format_table_headers(test_comp_table, test_widths, ["Evaluation Metric", "EfficientNet-B2 (Baseline)", "ConvNeXt-Tiny (Final Model)", "Net Absolute Gain"], bg_hex="1B4F72")

    test_rows = [
        ("Overall Test Accuracy", "70.85%", "73.43%", "+2.58% Improvement"),
        ("Macro Precision", "0.7192", "0.7394", "+0.0202"),
        ("Macro Recall", "0.7288", "0.7367", "+0.0079"),
        ("Macro F1-Score", "0.7121", "0.7240", "+0.0119 (+1.19 pts)"),
        ("Weighted F1-Score", "0.7090", "0.7276", "+0.0186 (+1.86 pts)")
    ]
    for row_idx, r in enumerate(test_rows, start=1):
        r_cells = test_comp_table.rows[row_idx].cells
        for col_idx in range(4):
            r_cells[col_idx].text = r[col_idx]
            set_cell_background(r_cells[col_idx], "FFFFFF" if row_idx % 2 == 0 else "F8F9F9")
            set_cell_margins(r_cells[col_idx], top=60, bottom=60, left=80, right=80)
            r_cells[col_idx].paragraphs[0].runs[0].font.size = Pt(9)
        r_cells[0].paragraphs[0].runs[0].font.bold = True
        r_cells[2].paragraphs[0].runs[0].font.bold = True

    # Embed Confusion Matrix Heatmap
    cm_img = os.path.join(results_dir, "confusion_matrix_heatmap.png")
    if os.path.exists(cm_img):
        doc.add_paragraph().paragraph_format.space_before = Pt(8)
        doc.add_picture(cm_img, width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 2: ConvNeXt-Tiny 17-Class Confusion Matrix Heatmap on Untouched Test Set.")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5)
        p_cap.runs[0].font.italic = True

    # Embed Per-Class F1 Bar Chart
    f1_img = os.path.join(results_dir, "per_class_f1_chart.png")
    if os.path.exists(f1_img):
        doc.add_paragraph().paragraph_format.space_before = Pt(8)
        doc.add_picture(f1_img, width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 3: Per-Class Test F1-Score Breakdown (Blue: Cattle Breeds | Brown: Buffalo Breeds).")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5)
        p_cap.runs[0].font.italic = True

    # =========================================================================
    # 7. IN-DEPTH ERROR ANALYSIS & MORPHOLOGICAL DRIVERS
    # =========================================================================
    doc.add_page_break()
    add_styled_heading(doc, "7. In-Depth Error Analysis & Morphological Drivers", level=1)

    doc.add_paragraph(
        "Out of 271 test images, ConvNeXt-Tiny made exactly 72 misclassifications (26.57% error rate). "
        "A rigorous audit revealed that errors were heavily clustered along phenotypic and morphological axes rather than random noise:"
    )

    doc.add_paragraph(
        "1. Buffalo-Buffalo Confusion (Primary Error Mode):\n"
        "   - Nagpuri -> Toda (7 errors): Nagpuri buffaloes misclassified as Toda due to dark coats and sweeping horns in outdoor shade.\n"
        "   - Murrah <-> Banni (7 errors bidirectional): Both breeds share jet-black coats and tightly coiled spiral horns. In cropped headshots where horn tips are obscured, visual disambiguation is near-impossible.\n"
        "   - Nili Ravi -> Murrah / Banni (4 errors): Riverine dairy conformation; misclassified when characteristic white facial blazes are occluded.\n\n"
        "2. Cattle-Cattle Confusion:\n"
        "   - Gir -> Krishna Valley (6 errors): Mottled red/grey coats under harsh sunlight confused with Krishna Valley grey-white coloration.\n"
        "   - Amritmahal <-> Hallikar (4 errors): Both belong to Karnataka's Mysore draught group, sharing upward-pointing horns and compact grey bodies.\n\n"
        "3. Cross-Species Errors (Extremely Rare):\n"
        "   - Only 6 out of 271 test images (2.2%) crossed the Cattle/Buffalo boundary, demonstrating that the single-stage 17-class classifier naturally learned robust species boundaries without requiring hierarchical cascading."
    )

    # Embed Prediction Gallery
    gallery_img = os.path.join(results_dir, "prediction_gallery.png")
    if os.path.exists(gallery_img):
        doc.add_paragraph().paragraph_format.space_before = Pt(6)
        doc.add_picture(gallery_img, width=Inches(6.2))
        p_cap = doc.add_paragraph("Figure 4: Representative Test Set Predictions (Top: Correct Classifications | Bottom: Morphological Misclassifications).")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5)
        p_cap.runs[0].font.italic = True

    # =========================================================================
    # 8. BACKEND & FRONTEND IMPLEMENTATION
    # =========================================================================
    doc.add_page_break()
    add_styled_heading(doc, "8. Full-Stack Implementation & API Architecture", level=1)

    add_styled_heading(doc, "8.1 Production Flask Backend (backend/app.py)", level=2)
    doc.add_paragraph(
        "The Flask service implements enterprise-grade ML serving patterns:\n"
        "- Singleton Architecture: Model weights are loaded and warmed up once at application startup.\n"
        "- Device Portability: Auto-detects NVIDIA CUDA GPU or multi-threaded CPU environments.\n"
        "- Input Validation: Enforces a 15 MB file size limit, validates image headers via PIL load integrity checks, "
        "and restricts input dimensions between 32x32 and 8192x8192 pixels.\n"
        "- REST Endpoints: GET /health (monitoring), GET /breeds (ICAR-NBAGR metadata), POST /predict (multipart file upload), "
        "and POST /predict-base64 (JSON base64 strings)."
    )

    add_styled_heading(doc, "8.2 React + Vite Frontend (cattle-sense-main)", level=2)
    doc.add_paragraph(
        "The user interface preserves the original aesthetic design while integrating full production features:\n"
        "- Real-Time HTML5 Webcam Scanner: Direct in-browser video feed with canvas frame grabber converting video streams to high-resolution JPEG blobs.\n"
        "- Species & Accession Badges: Clearly displays whether the detected animal is Cattle or Buffalo alongside ICAR accession IDs.\n"
        "- Ranked 17-Class Distribution: An interactive collapsible accordion displaying the complete softmax probability distribution across all 17 breeds.\n"
        "- Breeds Explorer: Live catalog connected to GET /breeds with dual filtering by species and agricultural utility."
    )

    # =========================================================================
    # 9. REPLICATION & DEPLOYMENT INSTRUCTIONS
    # =========================================================================
    add_styled_heading(doc, "9. Step-by-Step Execution Guide for Windows", level=1)

    doc.add_paragraph("Follow these exact commands to run the entire system on another Windows workstation:")

    doc.add_paragraph(
        "# 1. Clone Repository & Setup Virtual Environment\n"
        "git clone https://github.com/Doreshwar/SIH.git\n"
        "cd SIH\n"
        "python -m venv venv\n"
        ".\\venv\\Scripts\\activate\n"
        "pip install -r requirements.txt\n\n"
        "# 2. Build Frontend Production Assets\n"
        "cd cattle_breed_project\\cattle-sense-main\n"
        "npm install\n"
        "npm run build\n"
        "cd ..\\..\n\n"
        "# 3. Run Backend API Server (Terminal 1)\n"
        ".\\venv\\Scripts\\python cattle_breed_project\\backend\\app.py\n"
        "# (Server initializes ConvNeXt-Tiny on http://127.0.0.1:5000)\n\n"
        "# 4. Run Frontend Development Server (Terminal 2)\n"
        "cd cattle_breed_project\\cattle-sense-main\n"
        "npm run dev\n"
        "# (Frontend accessible at http://localhost:8080 or http://localhost:5173)\n\n"
        "# 5. Execute 37-Check Automated System Audit\n"
        ".\\venv\\Scripts\\python cattle_breed_project\\scripts\\run_end_to_end_audit.py"
    )

    # Save document
    doc.save(output_docx_path)
    doc.save(root_docx_path)
    print(f"Report successfully generated at:\n  1. {output_docx_path}\n  2. {root_docx_path}")

if __name__ == "__main__":
    main()
