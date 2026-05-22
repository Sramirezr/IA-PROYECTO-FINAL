from pathlib import Path

from fpdf import FPDF


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "data" / "sample_cvs"


CV_CASES = [
    (
        "cv_01_senior_dev.pdf",
        "Alex Johnson",
        "Senior Python Developer",
        [
            "6 years building backend services with Python, FastAPI, Docker, and AWS Lambda.",
            "Led migrations from monoliths to microservices using Kubernetes and CI/CD.",
            "AWS Solutions Architect Associate certification.",
        ],
    ),
    (
        "cv_02_financial_analyst.pdf",
        "Maria Garcia",
        "Financial Analyst",
        [
            "4 years in banking analytics at JP Morgan.",
            "Advanced Excel, SQL, dashboards, forecasting, and CFA Level 1 passed.",
            "Basic Python for financial reporting automation.",
        ],
    ),
    (
        "cv_03_nurse.pdf",
        "Emily Brown",
        "Registered Nurse ICU",
        [
            "5 years of ICU experience in a university hospital.",
            "Active RN license, BLS, ACLS, and CCRN certifications.",
            "Experience with mechanical ventilation and hemodynamic monitoring.",
        ],
    ),
    (
        "cv_04_marketing_manager.pdf",
        "David Smith",
        "Digital Marketing Manager",
        [
            "5 years managing Google Ads, Meta Ads, TikTok Ads, SEO, and GA4.",
            "Handled monthly media budgets up to 200k USD.",
            "Experience in campaign reporting and audience segmentation.",
        ],
    ),
    (
        "cv_05_civil_engineer.pdf",
        "Laura Miller",
        "Civil Engineer",
        [
            "Structural design, AutoCAD, Revit, site supervision, and budget control.",
            "Managed residential construction projects and contractor coordination.",
            "Knowledge of safety standards and project schedules.",
        ],
    ),
    (
        "cv_06_junior_analyst.pdf",
        "Kevin Lee",
        "Junior Data Analyst",
        [
            "Recent graduate with projects in SQL, Power BI, Python, and statistics.",
            "Built academic dashboards and cleaned survey datasets with pandas.",
            "Looking for an entry-level analytics role.",
        ],
    ),
    (
        "cv_07_teacher.pdf",
        "Sofia Martinez",
        "Primary School Teacher",
        [
            "7 years teaching math and reading to elementary students.",
            "Lesson planning, classroom management, and parent communication.",
            "Experience with inclusive education and formative assessment.",
        ],
    ),
    (
        "cv_08_accountant.pdf",
        "Robert Wilson",
        "Accountant",
        [
            "General ledger, reconciliation, tax filings, and monthly close processes.",
            "Experience with QuickBooks, SAP, Excel, and financial controls.",
            "Prepared audit support documentation for finance teams.",
        ],
    ),
    (
        "cv_09_graphic_designer.pdf",
        "Ana Torres",
        "Graphic Designer",
        [
            "Brand identity, social media assets, packaging, and editorial design.",
            "Advanced Adobe Illustrator, Photoshop, InDesign, and Figma.",
            "Portfolio includes campaigns for retail and hospitality clients.",
        ],
    ),
    (
        "cv_10_logistics.pdf",
        "Carlos Ruiz",
        "Logistics Coordinator",
        [
            "Warehouse operations, inventory control, routing, and supplier follow-up.",
            "Worked with ERP systems, KPIs, and last-mile delivery tracking.",
            "Reduced late shipments through process improvement initiatives.",
        ],
    ),
    (
        "cv_11_chef.pdf",
        "Julian Perez",
        "Chef",
        [
            "Kitchen leadership, menu planning, food safety, and cost control.",
            "Experience in hotels, restaurants, and high-volume catering.",
            "Managed purchasing and trained junior kitchen staff.",
        ],
    ),
    (
        "cv_12_history_grad.pdf",
        "Nora Evans",
        "History Graduate",
        [
            "Research, academic writing, archive analysis, and public presentations.",
            "Internship at a museum supporting cataloging and visitor education.",
            "Strong communication skills and document analysis.",
        ],
    ),
    (
        "cv_13_retail_seller.pdf",
        "Michael Davis",
        "Retail Sales Associate",
        [
            "Customer service, point-of-sale systems, merchandising, and inventory counts.",
            "Exceeded sales goals in apparel and consumer electronics stores.",
            "Experience handling returns and customer complaints.",
        ],
    ),
    (
        "cv_14_general_doctor.pdf",
        "Priya Patel",
        "General Practitioner",
        [
            "Primary care consultations, diagnosis, prescriptions, and patient follow-up.",
            "Experience in outpatient clinics and preventive health campaigns.",
            "Licensed physician with emergency triage knowledge.",
        ],
    ),
    (
        "cv_15_freshman.pdf",
        "Tom Walker",
        "First-Year Student",
        [
            "First-year computer science student with basic HTML and Python coursework.",
            "Volunteer experience in campus events and tutoring.",
            "Looking for an internship to build professional experience.",
        ],
    ),
]


def write_cv(filename: str, name: str, title: str, bullets: list[str]) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, name, ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, title, ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Profile", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for bullet in bullets:
        pdf.multi_cell(0, 7, f"- {bullet}")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Skills", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, "Communication, teamwork, problem solving, and professional discipline.")
    pdf.output(str(OUTPUT_DIR / filename))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in CV_CASES:
        write_cv(*case)
    print(f"Generated {len(CV_CASES)} sample CVs in {OUTPUT_DIR.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
