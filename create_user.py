from database import Session, User

session = Session()

new_user = User(username="admin", password="1234")

session.add(new_user)
session.commit()

print("User created successfully")