from pathlib import Path

DATA_DIR = Path.cwd() / 'data'


def main():
    notcar_dir = DATA_DIR / 'notcar'
    notcar_train = notcar_dir / 'train'
    notcar_test = notcar_dir / 'test'
    Path.mkdir(notcar_test, exist_ok=True)
    for seri in notcar_train.iterdir():
        image_list = []
        nofile = len(list(seri.glob('*')))
        target_seri = notcar_test / seri.name
        Path.mkdir(target_seri, exist_ok=True)
        count = 0
        for image in seri.iterdir():
            image.rename(target_seri / image.name)
            count += 1
            if count == nofile//2:
                break


if __name__ == "__main__":
    main()
