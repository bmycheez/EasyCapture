frame_cnt = 200
save_dir = './data_bright'
pipeline = [
    dict(type='GetRawFramefromCamera')]
camera = dict(
    fps = 30,
    gain = 300,
    exposure = 30)
