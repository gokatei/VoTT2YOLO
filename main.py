import sys
import os
import shutil
import random

import json
import yaml

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")

def outFolderInit(outputPath):
    if not(os.path.isdir(os.path.join(outputPath, 'images', 'train'))):
        os.makedirs(os.path.join(outputPath, 'images', 'train'))
        logging.debug('Create folder: '+os.path.join(outputPath, 'images', 'train'))
    if not(os.path.isdir(os.path.join(outputPath, 'images', 'val'))):
        os.makedirs(os.path.join(outputPath, 'images', 'val'))
        logging.debug('Create folder: '+os.path.join(outputPath, 'images', 'val'))

    if not(os.path.isdir(os.path.join(outputPath, 'labels', 'train'))):
        os.makedirs(os.path.join(outputPath, 'labels', 'train'))
        logging.debug('Create folder: '+os.path.join(outputPath, 'labels', 'train'))
    if not(os.path.isdir(os.path.join(outputPath, 'labels', 'val'))):
        os.makedirs(os.path.join(outputPath, 'labels', 'val'))
        logging.debug('Create folder: '+os.path.join(outputPath, 'labels', 'val'))

def getTagIndex(tags, tag):
    num = 0
    for tagData in tags:
        if tagData['name'] == tag:
            break
        num += 1
    return num

def main():
    if (len(sys.argv) >= 2):
        inputPath = sys.argv[1]
    if (len(sys.argv) >= 3):
        outputPath = sys.argv[2]

    imageExtensionList = [
        '.png',
        '.jpg',
        '.jpeg'
    ]
    inputPath = "./in"
    outputPath = "./out"

    print(inputPath)
    print(outputPath)

    if not(os.path.isdir(inputPath)):
        logging.error("入力パスを確認してください")
        exit()
    if not(os.path.isdir(outputPath)):
        logging.error("出力パスを確認してください")
        exit()

    jsonFileList = []
    imageList = []
    logging.info("Loading Files....")
    for fileName in os.listdir(inputPath):
        logging.debug(fileName)
        if os.path.isfile(os.path.join(inputPath, fileName)) & fileName.endswith('export.json'):
            jsonFileList.append(fileName)
        
        for imageExtension in imageExtensionList:
            if os.path.isfile(os.path.join(inputPath, fileName)) & fileName.endswith(imageExtension):
                imageList.append(fileName)
                
    if (len(jsonFileList) != 1):
        logging.error("入力ファイルを確認してください")
        exit()

    # 入力JSON読み込み
    annotationJSON = {}
    with open(os.path.join(inputPath, jsonFileList[0])) as jsonFile:
        annotationJSON = json.load(jsonFile)
        
    logging.info('Check output folder...')
    outputPath = os.path.join(outputPath, annotationJSON['name']+'-annotation')
    outFolderInit(outputPath) # 出力先フォルダの生成

    # 学習データ指定用yaml出力
    tagNames = []
    for tagData in annotationJSON['tags']:
        tagNames.append(tagData['name'])
    yamlData = {
        'train': '[train path hear]/images/train/',
        'val': '[val path hear]/images/val/',
        'nc': len(annotationJSON['tags']),
        'names': tagNames
    }
    with open(os.path.join(outputPath, annotationJSON['name']+'-annotation.yaml'), 'w') as yamlFile:
        yaml.dump(yamlData, yamlFile, encoding='utf-8', allow_unicode=True)

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
        shutil.copyfile(os.path.join(inputPath, asset['name']), os.path.join(outTrainImagePath, asset['name']))
        for region in annotationJSON['assets'][assetId]['regions']: # 各画像の指定範囲

            # yolo v7 移行のアノテーション規則に沿って計算
            # 画像の真ん中の座標と x y それぞれの幅を計算
            # https://qiita.com/yumi1123/items/ec023010ceeb05c2d73e 参考
            width = (region['points'][2]['x'] - region['points'][0]['x']) / asset['size']['width']
            height = (region['points'][2]['y'] - region['points'][0]['y']) / asset['size']['height']

            centerPointX = (region['points'][0]['x'] + (width / 2)) / asset['size']['width']
            centerPointY = (region['points'][0]['y'] + (height / 2)) / asset['size']['height']

            logging.debug(
                'Export Label text - ' +
                os.path.basename(asset['name']).split('.', 1)[0] + '.txt'+
                f"\n{centerPointX} {centerPointY} {width} {height}\n"
                )

            with open(os.path.join(outTrainLabelPath, os.path.basename(asset['name']).split('.', 1)[0] + '.txt'), mode='w') as labelFile:
                for tag in region['tags']: # 各画像の指定範囲に割り当てられたTAG(一個の指定範囲に複数TAGがあることもあり得る)
                    tagNum = getTagIndex(annotationJSON['tags'], tag) # tagが何番か
                    labelFile.write(f"{tagNum} {centerPointX} {centerPointY} {width} {height}\n")


    # 教師データエクスポート
    for assetId in annotationJSON['assets']:
        if (len(annotationJSON['assets'][assetId]['regions']) <= 0): # タグ指定ない画像は飛ばす
            continue
        if random.randrange(2) == 1: # 1/2 の確率でデータを取る
            continue
        asset = annotationJSON['assets'][assetId]['asset']
        logging.debug('Copy Image - ' + asset['name'])
        shutil.copyfile(os.path.join(inputPath, asset['name']), os.path.join(outValImagePath, asset['name']))
        for region in annotationJSON['assets'][assetId]['regions']: # 各画像の指定範囲

            # yolo v7 移行のアノテーション規則に沿って計算
            # 画像の真ん中の座標と x y それぞれの幅を計算
            # https://qiita.com/yumi1123/items/ec023010ceeb05c2d73e 参考
            width = (region['points'][2]['x'] - region['points'][0]['x']) / asset['size']['width']
            height = (region['points'][2]['y'] - region['points'][0]['y']) / asset['size']['height']

            centerPointX = (region['points'][0]['x'] + (width / 2)) / asset['size']['width']
            centerPointY = (region['points'][0]['y'] + (height / 2)) / asset['size']['height']

            logging.debug(
                'Export Label text - ' +
                os.path.basename(asset['name']).split('.', 1)[0] + '.txt'+
                f"\n{centerPointX} {centerPointY} {width} {height}\n"
                )

            with open(os.path.join(outValLabelPath, os.path.basename(asset['name']).split('.', 1)[0] + '.txt'), mode='w') as labelFile:
                for tag in region['tags']: # 各画像の指定範囲に割り当てられたTAG(一個の指定範囲に複数TAGがあることもあり得る)
                    tagNum = getTagIndex(annotationJSON['tags'], tag) # tagが何番か
                    labelFile.write(f"{tagNum} {centerPointX} {centerPointY} {width} {height}\n")

    logging.info('DONE')

if __name__ == "__main__": 
    logging.info("Start MAIN")
    main()