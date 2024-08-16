from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
import MySQLdb.cursors
import threading
import logging
import os
from flask import Flask, render_template, Response, request, jsonify, redirect
from flask_mysqldb import MySQL
import cv2
import numpy as np
import time

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set this to a unique and secret key

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'image_detection'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
mysql = MySQL(app)


@app.route('/', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        print(request.form)  # Print form data to the console
        email = request.form.get('email')
        password = request.form.get('mobileno')
        usertype = request.form.get('usertype')
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s AND mobileno = %s AND usertype = %s', (email, password, usertype))
        account = cursor.fetchone()

        if account:
            session['loggedin'] = True
            session['usertype'] = account['usertype']
            session['email'] = account['email']
            print(f"Usertype: {session['usertype']}")  # Debug print
            if session['usertype'] == 'superadmin':
                return redirect(url_for('superadmin'))
            elif session['usertype'] == 'branch':
                return redirect(url_for('header'))
        else:
            flash('Incorrect username/password!')

    return render_template('login.html')

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        state = request.form['state']
        name = request.form['name']
        usertype = request.form['usertype']
        email = request.form['email']
        mobileno = request.form['mobileno']
        
        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO users (state,  name, usertype, email, mobileno) VALUES ( %s, %s, %s, %s, %s)', 
                       (state,  name, usertype, email, mobileno))
        mysql.connection.commit()
        cursor.close()
        
        flash('User added successfully!', 'success')
        return redirect(url_for('user_list'))
    
    return render_template('user_master.html', users=[])

@app.route('/user_list', methods=['GET'])
def user_list():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()
    cursor.close()
    
    return render_template('user_master.html', users=users)
    return redirect(url_for('login_page'))





@app.route('/header')
def header():
    if 'loggedin' in session and session['usertype'] == 'branch':
        return render_template('header.html')
    return redirect(url_for('login_page'))

@app.route('/view_data')
def view_data():
    print("View Data Route Hit")
    cur = mysql.connection.cursor()
    cur.execute('SELECT * FROM image_detection.detection_count')
    data = cur.fetchall()
    cur.close()
    return render_template('view_data.html', data=data)

@app.route('/inverted')
def inverted():
    print("View Data Route Hit")
    cur = mysql.connection.cursor()
    cur.execute('SELECT * FROM image_detection.inverted_count')
    data = cur.fetchall()
    cur.close()
    return render_template('inverted1.html', data=data)

@app.route('/capture')
def capture():
    return render_template('n2.html')


@app.route('/front_image', methods=['POST'])
def front_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open video capture.")
        return jsonify({'status': 'error', 'message': 'Failed to open video capture'})

    ret, frame = cap.read()
    if ret:
        front_filename = 'front_image.jpg'
        front_filepath = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
        cv2.imwrite(front_filepath, frame)

        # Save image details to database
        cur = mysql.connection.cursor()
        cur.execute('INSERT INTO uploaded_images (front_filename, front_filepath) VALUES (%s, %s)', (front_filename, front_filepath))
        mysql.connection.commit()
        cur.close()

        cap.release()
        cv2.destroyAllWindows()
        return redirect('/capture')

    else:
        logging.error("Failed to capture front image.")
        cap.release()
        cv2.destroyAllWindows()
        return jsonify({'status': 'error', 'message': 'Failed to capture front image'}), 500

@app.route('/back_image', methods=['POST'])
def back_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open video capture.")
        return jsonify({'status': 'error', 'message': 'Failed to open video capture'})

    ret, frame = cap.read()
    if ret:
        back_filename = 'back_image.jpg'
        back_filepath = os.path.join(app.config['UPLOAD_FOLDER'], back_filename)
        cv2.imwrite(back_filepath, frame)

        # Save image details to database
        cur = mysql.connection.cursor()
        cur.execute('INSERT INTO uploaded_images (back_filename, back_filepath) VALUES (%s, %s)', (back_filename, back_filepath))
        mysql.connection.commit()
        cur.close()

        cap.release()
        cv2.destroyAllWindows()
        return redirect('/show_image')
    else:
        logging.error("Failed to capture back image.")
        cap.release()
        cv2.destroyAllWindows()
        return jsonify({'status': 'error', 'message': 'Failed to capture back image'}), 500

@app.route('/show_image')
def show_image():
    return render_template('n1.html')

@app.route('/front_crop', methods=['POST'])
def front_crop():
    try:
        front_data = request.json
        x = int(front_data['x'])
        y = int(front_data['y'])
        w = int(front_data['w'])
        h = int(front_data['h'])

        front_path = os.path.join(app.config['UPLOAD_FOLDER'], 'front_image.jpg')
        image = cv2.imread(front_path)

        if image is None:
            return jsonify({'status': 'error', 'message': 'Failed to load front image'}), 500

        height, width = image.shape[:2]
        
        # Validate coordinates
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        front_cropped = image[y:y+h, x:x+w]
        front_cropped_filename = 'front_cropped.jpg'
        front_cropped_filepath = os.path.join(app.config['UPLOAD_FOLDER'], front_cropped_filename)
        
        # Save cropped image
        if cv2.imwrite(front_cropped_filepath, front_cropped):
            cur = mysql.connection.cursor()
            cur.execute('INSERT INTO cropped_images (filename, filepath, side) VALUES (%s, %s, %s)', (front_cropped_filename, front_cropped_filepath, 'front'))
            mysql.connection.commit()
            cur.close()
            return jsonify({'status': 'success', 'message': 'Front cropped image saved', 'filepath': front_cropped_filepath}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save cropped image'}), 500
    except Exception as e:
        logging.error(f"Error during cropping front image: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@app.route('/back_crop', methods=['POST'])
def back_crop():
    try:
        back_data = request.json
        x = int(back_data['x'])
        y = int(back_data['y'])
        w = int(back_data['w'])
        h = int(back_data['h'])

        back_path = os.path.join(app.config['UPLOAD_FOLDER'], 'back_image.jpg')
        image = cv2.imread(back_path)

        if image is None:
            return jsonify({'status': 'error', 'message': 'Failed to load back image'}), 500

        height, width = image.shape[:2]

        # Validate coordinates
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        back_cropped = image[y:y+h, x:x+w]
        back_cropped_filename = 'back_cropped.jpg'
        back_cropped_filepath = os.path.join(app.config['UPLOAD_FOLDER'], back_cropped_filename)
        
        # Save cropped image
        if cv2.imwrite(back_cropped_filepath, back_cropped):
            cur = mysql.connection.cursor()
            cur.execute('INSERT INTO cropped_images (filename, filepath, side) VALUES (%s, %s, %s)', (back_cropped_filename, back_cropped_filepath, 'back'))
            mysql.connection.commit()
            cur.close()
            return jsonify({'status': 'success', 'message': 'Back cropped image saved', 'filepath': back_cropped_filepath}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save cropped image'}), 500
    except Exception as e:
        logging.error(f"Error during cropping back image: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@app.route('/set_inverted_count', methods=['POST'])
def set_inverted_count():
    inverted_count = int(request.form['inverted_count'])
    cur = mysql.connection.cursor()
    cur.execute('INSERT INTO inverted_count (count) VALUES (%s)', (inverted_count,))
    mysql.connection.commit()
    cur.close()
    return redirect('/show_image')



@app.route('/start_detection', methods=['POST'])
def start_detection():
    detection_thread = threading.Thread(target=run_detection)
    detection_thread.start()
    return jsonify({'status': 'detection started'}), 200

@app.route('/feed')
def feed():
    return render_template('feed.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_detection():
    with app.app_context():
        detect_and_count_image()

def detect_and_count_image():
    cur = mysql.connection.cursor()

    # Fetch the most recent cropped images
    cur.execute('SELECT filepath FROM cropped_images ORDER BY id DESC LIMIT 2')
    results = cur.fetchall()

    if not results:
        logging.error("No cropped images found in the database.")
        return

    # Load cropped images
    cropped_images = []
    for result in results:
        cropped_image_path = result[0]
        cropped_image = cv2.imread(cropped_image_path, cv2.IMREAD_GRAYSCALE)
        if cropped_image is not None:
            cropped_images.append((cropped_image_path, cropped_image))
        else:
            logging.error(f"Failed to load cropped image from {cropped_image_path}")

    if not cropped_images:
        logging.error("Failed to load any cropped images.")
        return

    # Fetch inverted count from the database
    cur.execute('SELECT count FROM inverted_count ORDER BY id DESC LIMIT 1')
    inverted_count_result = cur.fetchone()
    if inverted_count_result:
        inverted_count = inverted_count_result[0]
    else:
        logging.error("Inverted count not found in the database.")
        return

    # Open video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open video capture.")
        return

    frame_count = 0
    skip_frames = 1  # Remove frame skipping for faster detection
    detection_count = 0
    detection_interval = 0  # Set to 0 for continuous detection
    last_detection_time = time.time()  # Initialize the last detection time
    detected_boxes = []

    while detection_count <= inverted_count:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read frame from video capture.")
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (640, 480))
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        current_time = time.time()

        if current_time - last_detection_time >= detection_interval:
            found_any = False

            for target_image_path, target_image in cropped_images:
                target_w, target_h = target_image.shape[::-1]

                res = cv2.matchTemplate(gray_frame, target_image, cv2.TM_CCOEFF_NORMED)
                threshold = 0.4  # Adjust threshold if needed
                loc = np.where(res >= threshold)

                for pt in zip(*loc[::-1]):
                    x, y = pt
                    box = (x, y, x + target_w, y + target_h)

                    # Check if the box overlaps with any already detected boxes
                    overlap_found = False
                    for detected_box in detected_boxes:
                        if is_overlapping(box, detected_box):
                            overlap_found = True
                            break

                    if not overlap_found:
                        detected_boxes.append(box)

                        # Draw green box for new detections
                        cv2.rectangle(frame_resized, (x, y), (x + target_w, y + target_h), (0, 255, 0), 2)  # Green box

                        # Save detected region as image
                        cropped_image = frame_resized[y:y + target_h, x:x + target_w]
                        detected_filename = f'detected_region_{detection_count}.jpg'
                        detected_filepath = os.path.join(app.config['UPLOAD_FOLDER'], detected_filename)
                        if not cv2.imwrite(detected_filepath, cropped_image):
                            logging.error(f"Failed to save detected region image to {detected_filepath}")

                        # Insert detection data into the database
                        cur.execute('INSERT INTO detection_count (count, detected_filepath) VALUES (%s, %s)', (detection_count, detected_filepath))
                        mysql.connection.commit()

                        found_any = True
                        last_detection_time = current_time  # Update last detection time
                        detection_count += 1
                        break  # Stop checking other cropped images if one was found

            if not found_any:
                logging.debug("No matches found in the current frame.")

        cv2.imshow('Video Feed', frame_resized)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    cur.close()
    logging.info(f"Detection count: {detection_count}")

    return detection_count


def is_overlapping(box1, box2, threshold=0.3):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xa = max(x1, x3)
    ya = max(y1, y3)
    xb = min(x2, x4)
    yb = min(y2, y4)

    inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    overlap = inter_area / float(box1_area + box2_area - inter_area)

    return overlap > threshold

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            logging.error("Failed to capture frame from video feed.")
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)

@app.route('/superadmin')
def superadmin():
    if 'loggedin' in session and session['usertype'] == 'superadmin':
        return render_template('superadmin.html')
    return redirect(url_for('login_page'))



@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    session.pop('usertype', None)
    return redirect(url_for('login_page'))





if __name__ == '__main__':
    app.run(debug=True)
