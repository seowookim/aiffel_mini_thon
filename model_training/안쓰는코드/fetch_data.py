from pymongo import MongoClient

def fetch_documents(collection_name, limit=500):
    client = MongoClient('mongodb://mongodb:27017/')
    db = client['mini_thon']
    collection = db[collection_name]

    documents = list(collection.find().limit(limit))
    return documents

# train = fetch_documents('empathy_train')
# valid = fetch_documents('empathy_validation')
