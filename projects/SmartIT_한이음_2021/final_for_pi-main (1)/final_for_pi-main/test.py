import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Use a service account
cred = credentials.Certificate('abcd-27823-firebase-adminsdk-6hp4e-38c6900af9.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

user_ref = db.collection(u'hanium_db')
docs = user_ref.stream()
*_, last = docs




doc_ref = db.collection('hanium_db').document(f'{last.id}')
#get data from firestore
doc = doc_ref.get().to_dict()

print(last.id)
print(f'from databas: {doc}')
