- situation_id : 각 대화의 턴을 의미
- situation: 데이터의 "situation". 같은 대화 턴을 넣을 때는 situation은 다 중복
- speaker: "role": "listener"이면 1, "role": "speaker" 면 0
- change_emotion: "speaker_changeEmotion": null 이면 0, 아니면 1 
- empathy: "role":"listener"인 경우 "listener_empathy": [ ] 의 값.
  5가지의 경우가 있는데, [조언, 격려, 위로, 동조, null]. 각각 1,2,3,4,5로 코딩, role:speaker인 경우는 0
- speaker_emotion: ["기쁨", "당황", "분노", "불안", "상처", "슬픔", null] 의 경우가 있는데, 각각 1,2,3,4,5,6,0의 값을 가짐.
  처음에는 맨 위의 "speaker_emotion"에 해당하는 값을 가지고, 쭉쭉 똑같다가, 만약 change_emotion의 값이 1이면(즉, emotion 바뀌면) 값이 바뀜. 기쁨으로 바뀌면 1, 당황은 2, 불안은 3, 상처는 4, 슬픔은 5, 중립이나 그 외의 것들은 0.
- text:발화 내용 
- terminate: "terminate": true 일 때 1, 아니면 0. 즉 대화의 마지막이 1, 아니면 0 
