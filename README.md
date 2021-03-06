# Face_Detection_and_Recognition

## Real-time face recognition project with OpenCV and Python

### Introduction


What is face recognition? Or what is recognition? When you look at an apple fruit, your mind immediately tells you that this is an apple fruit. This process, your mind telling you that this is an apple fruit is recognition in simple words. So what is face recognition then? I am sure you have guessed it right. When you look at your friend walking down the street or a picture of him, you recognize that he is your friend Paulo. Interestingly when you look at your friend or a picture of him you look at his face first before looking at anything else. Ever wondered why you do that? This is so that you can recognize him by looking at his face. Well, this is you doing face recognition.

But the real question is how does face recognition works? It is quite simple and intuitive. Take a real life example, when you meet someone first time in your life you don't recognize him, right? While he talks or shakes hands with you, you look at his face, eyes, nose, mouth, color and overall look. This is your mind learning or training for the face recognition of that person by gathering face data. Then he tells you that his name is Paulo. At this point your mind knows that the face data it just learned belongs to Paulo. Now your mind is trained and ready to do face recognition on Paulo's face. Next time when you will see Paulo or his face in a picture you will immediately recognize him. This is how face recognition work. The more you will meet Paulo, the more data your mind will collect about Paulo and especially his face and the better you will become at recognizing him.

Now the next question is how to code face recognition with OpenCV, after all this is the only reason why you are reading this article, right? OK then. You might say that our mind can do these things easily but to actually code them into a computer is difficult? Don't worry, it is not. Thanks to OpenCV, coding face recognition is as easier as it feels. The coding steps for face recognition are same as we discussed it in real life example above.

#### Training Data Gathering: 
  Gather face data (face images in this case) of the persons you want to recognize.

#### Training of Recognizer:
  Feed that face data (and respective names of each face) to the face recognizer so that it can learn.

#### Recognition: 
  Feed new faces of the persons and see if the face recognizer you just trained recognizes them.

OpenCV comes equipped with built in face recognizer, all you have to do is feed it the face data. It's that simple and this how it will look once we are done coding it.

## LIBRARIES REQUIRED

<div class="highlight highlight-source-python"><pre><span class="pl-c">#import OpenCV module</span>
<span class="pl-k">import</span> <span class="pl-s1">cv2</span>
<span class="pl-c">#import os module for reading training data directories and paths</span>
<span class="pl-k">import</span> <span class="pl-s1">os</span>
<span class="pl-c">#import numpy to convert python lists to numpy arrays as </span>
<span class="pl-c">#it is needed by OpenCV face recognizers</span>
<span class="pl-k">import</span> <span class="pl-s1">numpy</span> <span class="pl-k">as</span> <span class="pl-s1">np</span>

<span class="pl-c">#matplotlib for display our images</span>
<span class="pl-k">import</span> <span class="pl-s1">matplotlib</span>.<span class="pl-s1">pyplot</span> <span class="pl-k">as</span> <span class="pl-s1">plt</span>
<span class="pl-c1">%</span><span class="pl-s1">matplotlib</span> <span class="pl-s1">inline</span> </pre></div>


<img style="-webkit-user-select: none;margin: auto;cursor: zoom-in;" src="https://raw.githubusercontent.com/Mjrovai/OpenCV-Face-Recognition/master/FaceRecogBlock.png" width="725" height="432">


