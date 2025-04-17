import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import firestore


class DBTransaction:
    def __init__(self) -> None:
        cred = credentials.Certificate(
            "./creds/firestore-semiotic-summer-423513-s2-afff6a1a3bd4.json"
        )
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def insert(self, doc, collection="users_feedback"):
        doc_ref = self.db.collection(collection).document()
        doc_ref.set(doc)
