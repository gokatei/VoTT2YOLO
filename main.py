import argparse
import sys
import os
import shutil
import random

import json
import yaml

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")

def outFolderInit(outputPath):
    pathList = {
        "images": [
            "train",
            "val"
        ],
        "labels": [
            "train",
            "val"
        ]
    }

    for pathName in pathList:
        for _pathName in pathList[pathName]:
            if not(os.path.isdir(os.path.join(outputPath, pathName, _pathName))):
                os.makedirs(os.path.join(outputPath, pathName, _pathName))
                logging.debug('Create folder: '+os.path.join(outputPath, pathName, _pathName))

def getTagIndex(tags, tag):
    num = 0
    for tagData in tags:
        if tagData['name'] == tag:
            break
        num += 1
    return num

def randomFloat(prob):
    if prob == False:
        return False
    if not 0 <= prob <= 1:
        return None
    if random.random() <= prob:
        return True
    else:
        return False


def main(args):
    imageExtensionList = [
        '.png',
        '.jpg',
        '.jpeg'
    ]

    if not(os.path.isdir(args.input)):
        logging.error('Please check input path.')
        exit()
    if not(os.path.isdir(args.output)):
        try:
            os.makedirs(args.output)
        except:
            logging.error('Please check output path.')
            exit()

    jsonFileList = []
    imageList = []
    logging.info("Loading Files....")
    for fileName in os.listdir(args.input):
        logging.debug(fileName)
        if os.path.isfile(os.path.join(args.input, fileName)) & fileName.endswith('export.json'):
            jsonFileList.append(fileName)
        
        for imageExtension in imageExtensionList:
            if os.path.isfile(os.path.join(args.input, fileName)) & fileName.endswith(imageExtension):
                imageList.append(fileName)
                
    if (len(jsonFileList) != 1):
        logging.error('Please check input files.')
        exit()

    # 入力JSON読み込み
    annotationJSON = {}
    with open(os.path.join(args.input, jsonFileList[0]), 'r', encoding="utf-8") as jsonFile:
        annotationJSON = json.load(jsonFile)
    
    if args.suffix: # 被らないようにサフィックス付けるやつ
        annotationJSON['name'] = annotationJSON['name'] + '-' + args.suffix
    
    logging.info('Check output folder...')
    outputPath = os.path.join(args.output, annotationJSON['name'])
    outFolderInit(outputPath) # 出力先フォルダの生成

    # 学習データ指定用yaml出力
    tagNames = []
    for tagData in annotationJSON['tags']:
        tagNames.append(tagData['name'])
    yamlData = {
        'train': f"./data/{annotationJSON['name']}/images/train/",
        'val': f"./data/{annotationJSON['name']}/images/val/",
        'nc': len(annotationJSON['tags']),
        'names': tagNames
    }
    with open(os.path.join(outputPath, 'data.yaml'), 'w', encoding="utf-8") as yamlFile:
        yaml.dump(yamlData, yamlFile, encoding='utf-8', allow_unicode=True)
    
    # 転移学習用スクリプト生成
    with open(os.path.join(outputPath, 'yolov7-train-start.sh'), 'w', encoding="utf-8", newline="\n") as startScript:
        startScript.write('# Please customize as needed!\n')
        startScript.write('cd ../../\n')
        startScript.write(
            "python3 train.py --workers 4 \\\n"+
            "    --device 0 \\\n"+
            "    --batch-size 32 \\\n"+
            f"    --data ./data/{annotationJSON['name']}/data.yaml \\\n"+
            "    --img 640 640 \\\n"+
            "    --cfg cfg/training/yolov7.yaml \\\n"+
            "    --weights 'yolov7_training.pt' \\\n"+
            f"    --name {annotationJSON['name']} \\\n"+
            "    --hyp data/hyp.scratch.custom.yaml \\\n"+
            "    --epochs 200"
            )
    with open(os.path.join(outputPath, 'yolov7-train-start.bat'), 'w', encoding="utf-8") as startScript:
        startScript.write('REM Please customize as needed!\n')
        startScript.write('cd ../../\n')
        startScript.write(
            "py train.py --workers 4 ^\n"+
            "    --device 0 ^\n"+
            "    --batch-size 32 ^\n"+
            f"    --data ./data/{annotationJSON['name']}/data.yaml ^\n"+
            "    --img 640 640 ^\n"+
            "    --cfg cfg/training/yolov7.yaml ^\n"+
            "    --weights 'yolov7_training.pt' ^\n"+
            f"    --name {annotationJSON['name']} ^\n"+
            "    --hyp data/hyp.scratch.custom.yaml ^\n"+
            "    --epochs 200"
            )

    # 画像やラベルのパス
    outTrainImagePath = os.path.join(outputPath, 'images', 'train')
    outTrainLabelPath = os.path.join(outputPath, 'labels', 'train')
    outValImagePath = os.path.join(outputPath, 'images', 'val')
    outValLabelPath = os.path.join(outputPath, 'labels', 'val')

    logging.info('Export Label texts and Copy Images...')

    # 学習データエクスポート
    for assetId in annotationJSON['assets']:
        if (len(annotationJSON['assets'][assetId]['regions']) <= 0): # タグ指定ない画像は飛ばす
            continue
        asset = annotationJSON['assets'][assetId]['asset']
        logging.debug('Copy Image - ' + asset['name'])

        outAssetName = asset['name']
        if args.imageSuffix:
            outAssetName = os.path.splitext(asset['name'])[0] + '-' + args.imageSuffix + os.path.splitext(asset['name'])[1]
        shutil.copyfile(os.path.join(args.input, asset['name']), os.path.join(outTrainImagePath, outAssetName))
        for region in annotationJSON['assets'][assetId]['regions']: # 各画像の指定範囲

            # yolo v7 移行のアノテーション規則に沿って計算
            # 画像の真ん中の座標と x y それぞれの幅を計算
            # https://qiita.com/yumi1123/items/ec023010ceeb05c2d73e 参考
            width = (region['points'][2]['x'] - region['points'][0]['x'])
            height = (region['points'][2]['y'] - region['points'][0]['y'])

            centerPointX = (region['points'][0]['x'] + (width / 2)) / asset['size']['width']
            centerPointY = (region['points'][0]['y'] + (height / 2)) / asset['size']['height']

            width = width / asset['size']['width']
            height =height / asset['size']['height']

            logging.debug(
                'Export Label text - ' +
                os.path.basename(outAssetName).split('.', 1)[0] + '.txt'+
                f"\n{centerPointX} {centerPointY} {width} {height}\n"
                )

            with open(os.path.join(outTrainLabelPath, os.path.basename(outAssetName).split('.', 1)[0] + '.txt'), 'w', encoding="utf-8") as labelFile:
                for tag in region['tags']: # 各画像の指定範囲に割り当てられたTAG(一個の指定範囲に複数TAGがあることもあり得る)
                    tagNum = getTagIndex(annotationJSON['tags'], tag) # tagが何番か
                    labelFile.write(f"{tagNum} {centerPointX} {centerPointY} {width} {height}\n")


    # 教師データエクスポート
    for assetId in annotationJSON['assets']:
        if (len(annotationJSON['assets'][assetId]['regions']) <= 0): # タグ指定ない画像は飛ばす
            continue
        if not randomFloat(args.valPercent): # 指定した確率で出力
            continue
        asset = annotationJSON['assets'][assetId]['asset']
        logging.debug('Copy Image - ' + asset['name'])

        outAssetName = asset['name']
        if args.imageSuffix:
            outAssetName = os.path.splitext(asset['name'])[0] + '-' + args.imageSuffix + os.path.splitext(asset['name'])[1]
        shutil.copyfile(os.path.join(args.input, asset['name']), os.path.join(outValImagePath, outAssetName))
        for region in annotationJSON['assets'][assetId]['regions']: # 各画像の指定範囲

            # yolo v7 移行のアノテーション規則に沿って計算
            # 画像の真ん中の座標と x y それぞれの幅を計算
            # https://qiita.com/yumi1123/items/ec023010ceeb05c2d73e 参考
            width = (region['points'][2]['x'] - region['points'][0]['x'])
            height = (region['points'][2]['y'] - region['points'][0]['y'])

            centerPointX = (region['points'][0]['x'] + (width / 2)) / asset['size']['width']
            centerPointY = (region['points'][0]['y'] + (height / 2)) / asset['size']['height']
            
            width = width / asset['size']['width']
            height = height / asset['size']['height']

            logging.debug(
                'Export Label text - ' +
                os.path.basename(outAssetName).split('.', 1)[0] + '.txt'+
                f"\n{centerPointX} {centerPointY} {width} {height}\n"
                )

            with open(os.path.join(outValLabelPath, os.path.basename(outAssetName).split('.', 1)[0] + '.txt'), 'w', encoding="utf-8") as labelFile:
                for tag in region['tags']: # 各画像の指定範囲に割り当てられたTAG(一個の指定範囲に複数TAGがあることもあり得る)
                    tagNum = getTagIndex(annotationJSON['tags'], tag) # tagが何番か
                    labelFile.write(f"{tagNum} {centerPointX} {centerPointY} {width} {height}\n")

    logging.info('DONE')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VoTT2YOLO')
    parser.add_argument('input', help='Set input directory')
    # parser.add_argument('-i', '--input', help='Set input directory', default='./input')
    parser.add_argument('-o', '--output', help='Set output directory', default='./output')
    parser.add_argument('-v', '--valPercent', help='Validation Percentage - Specified float of between "0-1". Disable the output of validation data with "false" or "0".', default=0.3)
    parser.add_argument('-s', '--suffix', help='Add a suffix to avoid duplication of saves.')
    parser.add_argument('-is', '--imageSuffix', help='Add a suffix to images and labels.')
    args = parser.parse_args()

    if (args.valPercent == 'false'):
        args.valPercent = False
    else:
        args.valPercent = float(args.valPercent)
    if (randomFloat(args.valPercent)) == None:
        logging.error('Please input valPercent is float of between 0-1.')
        exit()
    logging.debug(args)
    logging.debug("Start MAIN")
    main(args)