"""
Seed script to create an admin user.

Usage:
    python seed_admin.py

Default admin credentials:
    Username: admin
    Email: admin@heartpredict.com
    Password: Admin@123 (change this in production!)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import app, db
from models import User
from auth import hash_password

def create_admin():
    with app.app_context():
        admin = User.query.filter_by(username='admin').first()
        if admin:
            print("Admin user already exists.")
            return

        admin = User(
            username='admin',
            email='admin@heartpredict.com',
            password_hash=hash_password('Admin@123'),
            role='admin',
            status='active'
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin user created successfully!")
        print("Username: admin")
        print("Email: admin@heartpredict.com")
        print("Password: Admin@123")
        print("\nIMPORTANT: Change this password immediately in production!")

if __name__ == "__main__":
    create_admin()