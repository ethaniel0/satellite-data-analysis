import numpy as np
from PIL import Image, ImageEnhance
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
import rasterio
from rasterio.windows import Window
import random
import pickle
from shapely import geometry
from matplotlib import colors as mcolors
from random import shuffle
import cv2
from collections import OrderedDict
from scipy.ndimage import convolve

class Classifier:
    def __init__(self, bandset):
        self.classes = OrderedDict()
        self.bandset = bandset
        self.bigimg = None

        self.bigimg = self.combineLayers(self.bandset, range(len(self.bandset)))
    
    def save_training(self, filename):
        pickle.dump(self.classes, open(filename, "wb"))
    
    def load_training(self, data):
        d = pickle.load(open(data, "rb"))
        for key, value in d.items():
            self.classes[key] = value

    def combineLayers(self, bandset, bandorder):
        img = [self.bandset[band].read(1) for band in bandorder]

        maxhi, maxwid = max(img, key=lambda x: x.shape[1]).shape

        for i in range(len(img)):
            img[i] = cv2.resize(img[i], (maxwid, maxhi))
        
        return np.dstack(img)

    # add a class to the classifier
    def add(self, name, polygons):
        avgvalues = []
        # go through each band in the bandset, read it, and get the average value for every polygon through that band
        for bandimg in self.bandset:
            band = bandimg.read(1)
            averages = []
            for polygon in polygons:
                points = []
                for point in polygon:
                    # all the points are normalized, so scale them back up
                    points.append((round(point[0] * band.shape[1]), round(point[1] * band.shape[0])))
                avg = self.getAverageColor(band, points)
                averages.append(avg)
            avgvalues.append(np.average(averages, axis=0))
        self.classes[name] = np.array(avgvalues)

    def getBoundingSquare(self, points):
        ps = np.array(points)
        xs = ps[::,0]
        ys = ps[::,1]
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)
        return (xmin, xmax, ymin, ymax)

    def getNearbyPixels(self, x, y, radius=2):
        xmin = max(0, x - radius)
        xmax = min(self.bigimg.shape[1], x + radius)
        ymin = max(0, y - radius)
        ymax = min(self.bigimg.shape[0], y + radius)
        pixels = self.bigimg[ymin:ymax, xmin:xmax].flatten().tolist()
        # pad with zeroes if not long enough
        if len(pixels) != (2 * radius + 1) ** 2 * self.bigimg.shape[2]:
            pads = ((2 * radius + 1) ** 2 - len(pixels))
            pixels.extend([[0]*13 for _ in range(pads)])
        return pixels

    def getPixelsWithinPolygon(self, img, points):
        xmin, xmax, ymin, ymax = self.getBoundingSquare(points)
        polygon = geometry.polygon.Polygon(points)
        pixels = []
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                if polygon.contains(geometry.Point(x, y)):
                    pixels.append(img[y][x])
        return pixels
    
    def getAverageColor(self, img, points):
        pixels = self.getPixelsWithinPolygon(img, points)
        return np.average(pixels, axis=0)
    
    def getPixelCoords(self, points):
        xmin, xmax, ymin, ymax = self.getBoundingSquare(points)
        polygon = geometry.polygon.Polygon(points)
        pixels = []
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                if polygon.contains(geometry.Point(x, y)):
                    pixels.append(self.getNearbyPixels(x, y))
        return pixels

    def nearestClass(self, x, y, radius=2):
        if radius > 0:
            xmin = max(0, x - radius)
            xmax = min(self.bigimg.shape[1], x + radius)
            ymin = max(0, y - radius)
            ymax = min(self.bigimg.shape[0], y + radius)
            pixel = np.average(self.bigimg[ymin:ymax, xmin:xmax], axis=(0,1))
        else:
            pixel = self.bigimg[y][x]
        
        colors = np.array(list(self.classes.values()))
        dists = np.linalg.norm(colors - pixel, axis=1)
        return np.argmin(dists)
    
    def nearestVectorizeClass(self, pixel, colors, possiblecolors):
        return possiblecolors[np.argmin(np.linalg.norm(colors - pixel, axis=1))]
    
    def classifyRegion(self, x1, y1, x2, y2, colors, verbose=False, radius=2):
        results = np.zeros((y2 - y1, x2 - x1, 3), 'uint8')
        
        possibleColors = []
        colorkeys = list(colors.keys())
        classkeys = list(self.classes.keys())
        colorValues = np.array(list(self.classes.values()))
        colorkeys.sort(key=lambda x: classkeys.index(x) if x in classkeys else 1000)
        for key in colorkeys:
            possibleColors.append(np.round(np.array(mcolors.to_rgb(colors[key])[:3]) * 255).astype('uint8'))
        colorkeys = np.array(colorkeys)

        numcolors = self.bigimg.shape[2]
        kernel = np.ones((2*radius + 1,2*radius + 1), dtype=np.float64) / (2*radius + 1)**2
        kernel = kernel[:, :, None]
        averaged = convolve(self.bigimg[y1:y2,x1:x2].astype(np.float64), kernel, mode='reflect')
        # averaged -= colorValues.astype(np.float64)

        # func = lambda p: possibleColors[np.argmin(np.linalg.norm(colorValues - p, axis=1))]

        # nearfunc = np.vectorize(func, signature='(10)->(3)')
        # return nearfunc(averaged[y1:y2, x1:x2])

        checkpoint = (y2-y1) // 10
        for y in range(y2 - y1):
            for x in range(x2 - x1):
                classname = np.argmin(np.linalg.norm(colorValues - averaged[y][x], axis=1))
                results[y][x] = possibleColors[classname]
            if verbose:
                if y % checkpoint == 0:
                    print(str(round(y / (y2 - y1) * 100)) + '%')
        
        return np.array(results)
        
        # checkpoint = (y2-y1) // 10
        # for y in range(y1, y2):
        #     for x in range(x1, x2):
        #         classname = self.nearestClass(x, y, radius)
        #         results[y-y1][x-x1] = possibleColors[classname]
        #     if verbose:
                
        #         if (y - y1) % checkpoint == 0:
        #             print(str(round((y - y1) / (y2 - y1) * 100)) + '%')
        
        # return np.array(results)
    
    def classify(self, colors, verbose=False, radius=2):
        return self.classifyRegion(0, 0, self.bigimg.shape[1], self.bigimg.shape[0], colors, verbose, radius)

class EGIS:
    def __init__(self, dir, bands=('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12')):
        self.bands = bands
        self.bandset = []
        self.bandpaths = []

        self.areas = {}
        self.macroclasses = []
        self.classes = {}
        self.colors = {}
        self.colorMap = {}

        self.colorAreas = {}

        self.preview = None

        self.classifier = None
        self.classification = None

        self.load_bands(dir)
    
    def load_bands(self, directory):
        self.bandset = []
        self.bandpaths = []
        for file in os.listdir(directory):
            if file.endswith(".jp2") and file.split('.jp2')[0].split('_')[-1] in self.bands:
                self.bandpaths.append(directory + file)
                # bandset.append(rasterio.open(dir + file))
        self.bandpaths.sort()
        for band in self.bandpaths:
            self.bandset.append(rasterio.open(band))
        
        self.classifier = Classifier(self.bandset)
    
    def show_bands(self, skip=4):
        f, axarr = plt.subplots(3,4)
        for ind, band in enumerate(self.bandset):
            axarr[ind//4][ind%4].imshow(band.read(1)[::skip, ::skip])
    
    def getCropArea(self):
        fig, ax = plt.subplots()
        img = self.bandset[0]
        plt.imshow(img.read(1))
        self.cropArea = {
            'tl': [],
            'br': [],
            'tlcoords': [],
            'brcoords': []
        }
        def onclick(event):
            x, y = event.xdata, event.ydata
            rightclick = event.button == 3
            if rightclick:
                self.cropArea['br'] = [x / img.shape[1], y / img.shape[0]]
                self.cropArea['brcoords'] = [round(x), round(y)]
                print('bottom right', self.cropArea['brcoords'])
            else:
                self.cropArea['tl'] = [x / img.shape[1], y / img.shape[0]]
                self.cropArea['tlcoords'] = [round(x), round(y)]
                print('top left', self.cropArea['tlcoords'])
            if self.cropArea['tl'] and self.cropArea['br']:
                x = self.cropArea['tlcoords'][0]
                y = self.cropArea['tlcoords'][1]
                wid = self.cropArea['brcoords'][0] - x
                hi = self.cropArea['brcoords'][1] - y
                ax.add_patch(Rectangle((x, y), wid, hi, fill=False, edgecolor='red'))
                plt.draw()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    
    def saveCrops(self, clipFolder='clipped/', clipname='clip-'):
        for path in self.bandpaths:
            with rasterio.open(path) as src:
                # Create a Window and calculate the transform from the source dataset
                wid = src.shape[1]
                hi = src.shape[0]
                xsize = round((self.cropArea['br'][0] - self.cropArea['tl'][0]) * wid)
                ysize = round((self.cropArea['br'][1] - self.cropArea['tl'][1]) * hi)
                window = Window(round(self.cropArea['tl'][0] * wid), round(self.cropArea['tl'][1] * hi), round(xsize * wid), round(ysize * hi))
                transform = src.window_transform(window)
                profile = src.profile
                profile.update({
                    'height': ysize,
                    'width': xsize,
                    'transform': transform
                })
                
                with rasterio.open(clipFolder + clipname +  path.split('/')[-1], 'w', **profile) as dst:
                    # Read the data from the window and write it to the output raster
                    dst.write(src.read(window=window))
                    
                print(clipFolder + clipname + path.split('/')[-1])
    
    def combineLayers(self, bandorder):
        img = [self.bandset[band].read(1) for band in bandorder]

        maxhi, maxwid = max(img, key=lambda x: x.shape[1]).shape

        for i in range(len(img)):
            img[i] = cv2.resize(img[i], (maxwid, maxhi))
        
        return np.dstack(img)
    
    def getRGBImage(self, bandorder, darkness=32, enhanceColor=2, enhanceBrightness=1.5):
        img = self.combineLayers(bandorder)

        img = img / darkness

        # make sure that the whites that are too bright don't overflow
        bounds = lambda x: max(0, min(x, 255))
        vfunc = np.vectorize(bounds)
        img = vfunc(img)
        img = img.astype(np.uint8)

        previewimg = Image.fromarray(img, mode='RGB')
        addColor = ImageEnhance.Color(previewimg)
        pimg2 = addColor.enhance(enhanceColor)
        brighten = ImageEnhance.Brightness(pimg2)
        pimg2 = brighten.enhance(enhanceBrightness)
        self.preview = np.array(pimg2)

        return self.preview
    
    def setClasses(self, macroclasses, classes, colors):
        self.macroclasses = macroclasses
        self.classes = classes
        self.colors = colors
        for key in classes:
            for val in classes[key]:
                rgb = mcolors.to_rgb(colors[val])[:3]
                self.colorMap[tuple(np.round(np.array(rgb)*255).astype(np.uint8).tolist())] = val
    
    def selectROIs(self):
        fig, ax = plt.subplots()
        ax.imshow(self.getRGBImage((2, 1, 0)))

        cur_macroclass = 0
        cur_class = 0

        coords = []             # normalized coordinates because some images are larger than others
        pixelcoords = []        # coordinates in pixels, relative to the reference image

        currentcircles = []     # holds the circles drawn on the image

        pressingShift = False

        curtext = None
        curpolygon = None

        mcstr = self.macroclasses[cur_macroclass] # macro class name
        curtext = plt.text(0, 0.94, mcstr + ',' + self.classes[mcstr][cur_class], color='red', transform = ax.transAxes)
        fig.canvas.draw()

        def onclick(event):
            if not pressingShift: return
            nonlocal coords, currentcircles, pixelcoords, curpolygon
            x, y = event.xdata, event.ydata
            rightclick = event.button == 3

            mcname = self.macroclasses[cur_macroclass]
            cname = self.classes[mcname][cur_class]
            
            # close the polygon, reset coordinates
            if rightclick:
                # add the coordinates to the areas
                
                if not mcname in self.areas:
                    self.areas[mcname] = []
                if not cname in self.areas:
                    self.areas[cname] = []
                
                self.areas[mcname].append(coords)
                self.areas[cname].append(coords)
                

                coords = []
                pixelcoords = []
                for i in currentcircles:
                    i.remove()
                fig.canvas.draw()
                currentcircles = []
                curpolygon = None
            # add coordinate, draw point
            else:
                coords.append((x / self.preview.shape[1], y / self.preview.shape[0]))
                pixelcoords.append((round(x), round(y)))
                circle = Circle((round(x), round(y)), radius=3, fill=True, alpha=0.8, color='magenta')
                ax.add_patch(circle)
                currentcircles.append(circle)
            
            # draw a polygon if more than 3 coordinates
            if len(coords) > 2:
                if curpolygon: curpolygon.remove()
                curpolygon = Polygon(np.array(pixelcoords), fill=True, alpha=0.6, color=self.colors[cname])
                ax.add_patch(curpolygon)
            
            fig.canvas.draw()

        def onKeyPress(event):
            nonlocal pressingShift
            if event.key == 'shift':
                pressingShift = True

        def onKeyUp(event):
            nonlocal pressingShift, cur_macroclass, cur_class, curtext, curpolygon
            if event.key == 'shift':
                pressingShift = False
            if event.key == '1':
                cur_macroclass = max(0, cur_macroclass - 1)
                cur_class = 0
            if event.key == '2':
                cur_macroclass = min(len(self.macroclasses) - 1, cur_macroclass + 1)
                cur_class = 0
            if event.key == '3':
                cur_class = max(0, cur_class - 1)
            if event.key == '4':
                cur_class = min(len(self.classes[self.macroclasses[cur_macroclass]]) - 1, cur_class + 1)

            mcstr = self.macroclasses[cur_macroclass]
            if curtext: curtext.remove()
            curtext = plt.text(0, 0.94, mcstr + ',' + self.classes[mcstr][cur_class], color='red', transform = ax.transAxes)
            fig.canvas.draw()
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        keydown = fig.canvas.mpl_connect('key_press_event', onKeyPress)
        keyup = fig.canvas.mpl_connect('key_release_event', onKeyUp)
    
    def saveAreas(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.areas, f)
    
    def loadAreas(self, filename):
        with open(filename, 'rb') as f:
            self.areas = pickle.load(f)
    
    def loadClassifierTraining(self, filename):
        self.classifier.load_training(filename)
    
    def saveClassifierTraining(self, filename):
        self.classifier.save_training(filename)
    
    def addClassesToClassifier(self):
        for key in self.classes:
            for x in self.classes[key]:
                if x in self.areas and x not in self.classifier.classes:
                    self.classifier.add(x, self.areas[x])
                    print(x)
    
    def previewClassifier(self):
        if self.preview is None:
            self.getRGBImage((4, 3, 2))
        fig, ax = plt.subplots()
        ax.imshow(self.preview)

        def previewOnclick(event):
            x, y = event.xdata, event.ydata
            xmin = max(0, int(x - 200))
            xmax = min(self.preview.shape[1], int(x + 200))
            ymin = max(0, int(y - 200))
            ymax = min(self.preview.shape[0], int(y + 200))
            classification = self.classifier.classifyRegion(xmin, ymin, xmax, ymax, self.colors, verbose=True, radius=1)
            ax.imshow(classification)
            fig.canvas.draw()

        cid = fig.canvas.mpl_connect('button_press_event', previewOnclick)
    
    def generateClassification(self):
        classification = self.classifier.classify(self.colors, verbose=True, radius=1)
        self.classification = classification
        for i in self.classification:
            for j in i:
                c = self.colorMap[tuple(j.tolist())]
                if c not in self.colorAreas:
                    self.colorAreas[c] = 0
                self.colorAreas[c] += 1
        return classification

    def accuracyPoints(self, k=10):
        colorPixels = {}

        totalPixels = sum(list(self.colorAreas.values()))

        if self.classification is None:
            self.generateClassification()

        for y in range(len(self.classification)):
            for x in range(len(self.classification[y])):
                color = tuple(self.classification[y][x])
                if color not in colorPixels:
                    colorPixels[color] = []
                colorPixels[color].append((x, y))

        testpoints = []
        for key in colorPixels:
            points = random.choices(colorPixels[key], k=k)
            testpoints.append(points)
        
        fig, axarr = plt.subplots()
        axarr.imshow(self.preview)

        curclass = 0
        curpoint = 0
        curtext = None
        cursquare = None

        self.failed = 0
        self.passed = 0
        self.perClass = {}


        point = testpoints[curclass][curpoint]
        predicted = tuple(self.classification[point[1]][point[0]].tolist())
        colorname = self.colorMap[predicted]

        cursquare = axarr.add_patch(Rectangle((point[0] - 3, point[1] - 3), 6, 6, edgecolor='red', facecolor='none'))
        curtext = plt.text(0, 0.94, colorname, color='red', transform = axarr.transAxes)

        proportion = self.colorAreas[colorname] / totalPixels

        radius = 300

        # y axis goes from top down, so max and min are reversed
        axarr.axis([
                    max(0, point[0]-radius),                  # x min
                    min(self.preview.shape[1], point[0]+radius),   # x max
                    min(self.preview.shape[0], point[1]+radius),   # y max
                    max(0, point[1]-radius)                   # y min
                    ])

        self.perClass[colorname] = { 'passed': 0, 'failed': 0 }

        def accuracyKeyPress(event):
            nonlocal cursquare, curtext, curpoint, curclass, curtext, colorname, proportion
            if colorname not in self.perClass:
                self.perClass[colorname] = {
                    'passed': 0,
                    'failed': 0
                }
            # failed
            if event.key == '1':
                self.failed += proportion
                self.perClass[colorname]['failed'] += 1
                print('failed', end=' ')
            
            # passed
            if event.key == '2':
                self.passed += proportion
                self.perClass[colorname]['passed'] += 1
                print('passed', end=' ')
            
            curpoint += 1
            if curpoint >= len(testpoints[curclass]):
                curclass += 1
                curpoint = 0
                print()
            
            # define new point and color, remove old markers before replacing them
            point = testpoints[curclass][curpoint]
            predicted = tuple(self.classification[point[1]][point[0]].tolist())
            colorname = self.colorMap[predicted]
            cursquare.remove()
            curtext.remove()
            cursquare = axarr.add_patch(Rectangle((point[0] - 3, point[1] - 3), 6, 6, edgecolor='red', facecolor='none'))
            curtext = plt.text(0, 0.94, colorname, color='red', transform = axarr.transAxes)
            proportion = self.colorAreas[colorname] / totalPixels
            
            axarr.axis([max(0, point[0]-radius), min(self.preview.shape[1], point[0]+radius), min(self.preview.shape[0], point[1]+radius), max(0, point[1]-radius)])
            fig.canvas.draw()

        keydown = fig.canvas.mpl_connect('key_press_event', accuracyKeyPress)
    
    def accuracyAssessment(self):
        print(str(100*self.passed / (self.passed + self.failed)) + "% Accuracy")
        for key in self.perClass:
            print(key + ': ' + str(100*self.perClass[key]['passed'] / 10) + "% Accuracy")