# Description: MongoDB 연결을 확인하는 스크립트
# http://localhost:27017/로 접속했을때 text가 나온다면 실행되고 있다는 뜻인데, db 연결이 안되서 테스트하는 코드를 따로 만듦

from pymongo import MongoClient

def check_mongo_connection():
    try:
        # MongoDB 클라이언트 설정 (인증 정보 없음)
        client = MongoClient('mongodb://mongodb:27017/')
        db = client['mini_thon']
        collection = db['conversations']
        
        # 데이터베이스와 컬렉션에 접근 시도
        db_names = client.list_database_names()
        print("Databases:", db_names)

        # 컬렉션 목록을 가져옴
        collection_names = db.list_collection_names()
        print("Collections in 'mini_thon' database:", collection_names)

        print("MongoDB connection successful.")
        
    except Exception as e:
        print("Failed to connect to MongoDB:", e)

if __name__ == "__main__":
    check_mongo_connection()
