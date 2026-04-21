# src/itsm_agent/agent/registry.py

DEPARTMENT_REGISTRY: dict[str, dict] = {
    "IT Software": {
        "description": "Issues with installed applications, OS updates, and developer tools.",
        "keywords": ["Git", "IntelliJ", "VS Code", "Python", "Java", "Slack", "Office 365"],
    },
    "DT-GPS": {
        "description": "Global Provisioning Services. Primary silo for access and identity.",
        "keywords": ["Okta", "VPN", "SSO", "Zscaler", "Login", "Password Reset", "Access Denied"],
    },
    "Global People Live Chat Agents": {
        "description": "Human Resources and employee perks.",
        "keywords": ["Matching Gifts", "Benefits", "Gym Reimbursement", "Payroll", "HR"],
    },
    "WPS - Badging": {
        "description": "Workplace Services and physical office facilities.",
        "keywords": ["Badge access", "Office temp", "Desk booking", "Physical Maintenance"],
    },
}
