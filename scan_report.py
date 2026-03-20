import fitz  # PyMuPDF
import os

def create_and_scan():
    filename = "test_report.pdf"
    
    # 1. Create a dummy medical report if one doesn't exist
    if not os.path.exists(filename):
        print(f"📝 Creating a dummy medical report: {filename}")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Patient Name: Rudra Rana\nDiagnosis: Malignant Mass found.\nBI-RADS Category: 5\nCalcification: Present.")
        doc.save(filename)
        doc.close()

    # 2. Now, let's SCAN it
    print(f"🔍 Scanning: {filename}...")
    doc = fitz.open(filename)
    report_text = ""
    for page in doc:
        report_text += page.get_text()
    
    # 3. Look for the "Red Flags"
    red_flags = ["Malignant", "BI-RADS", "Mass", "Calcification"]
    
    print("\n--- 🩺 Scan Results ---")
    for flag in red_flags:
        if flag.lower() in report_text.lower():
            print(f"🚩 FOUND: {flag}")
        else:
            print(f"✅ NOT FOUND: {flag}")

if __name__ == "__main__":
    create_and_scan()
