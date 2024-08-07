# AI hub의 공감형 대화를 다운로드 받아서 MongoDB에 데이터를 삽입하는 코드
    # https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71305
    # API를 쓸 수도 있으나, 시간상 직접 다운로드 받아서 사용
# 수작업이 필요한 과정임

# ------------------------------------------------------------------ #
#%%
import zipfile
import os

# 압축을 해제할 ZIP 파일이 있는 디렉토리 경로
# directory_path = './data/046.공감형 대화/01-1.정식개방데이터/Training/02.라벨링데이터'
directory_path = './data/046.공감형 대화/01-1.정식개방데이터/Validation/02.라벨링데이터'

# 압축을 해제할 경로
# output_directory = './data/train'
output_directory = './data/validation'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 디렉토리 내의 모든 파일을 반복
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.zip'):
            zip_file_path = os.path.join(root, file)
            # output_directory = os.path.join(root, os.path.splitext(file)[0])

            # 출력 디렉토리 생성
            os.makedirs(output_directory, exist_ok=True)

            # ZIP 파일 열기 및 압축 해제
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(output_directory)

            print(f'ZIP 파일 {zip_file_path}의 압축을 {output_directory}에 해제했습니다.')

# ------------------------------------------------------------------ #
#%%
import os
import json
from pymongo import MongoClient

# MongoDB 클라이언트 설정
# compose 파일에서 environment에 설정한 사용자 이름과 비밀번호를 사용하니 error 111 connection refused 발생
# 이유는 못찾음, 4시간을 삽질하다가 environment를 주석처리하고 (아무나 접속 가능하게 하고) 해결
client = MongoClient('mongodb://mongodb:27017/')
db = client['mini_thon']
collection = db['empathy_train']
# collection = db['empathy_validation']

# JSON 파일이 저장된 디렉토리
json_directory = './data/train'
# json_directory = './data/validation'

def insert_json_to_mongo(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            collection.insert_one(data)  # 데이터 삽입
            print(f'{json_file} has been inserted into MongoDB')
    except Exception as e:
        print(f'Failed to insert {json_file} into MongoDB: {e}')

# 디렉토리 내의 모든 JSON 파일 처리
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        file_path = os.path.join(json_directory, filename)
        insert_json_to_mongo(file_path)

    


