import streamlit as st
from PIL import Image
import cv2
import numpy as np
import easyocr


def find_document_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        if len(approx) == 4:
            return approx

    return None


def reorder_contour_points(pts):
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def warp_perspective(image, document_contour):
    ordered_points = reorder_contour_points(document_contour)

    (tl, tr, br, bl) = ordered_points

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_points, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


# 缩放比例
re_size = 0.5


# 主要算法
def sign_analysis(pic):
    photo = pic

    # 寻找文档轮廓
    document_contour = find_document_contour(photo)

    if document_contour is not None:
        # 校正图片
        result = warp_perspective(photo, document_contour)
    else:
        print("未找到文档轮廓")
        exit()

    # 进行OCR识别,判断签名的位置
    reader = easyocr.Reader(['ch_sim', 'en'])  # 这里你可以根据需要选择语言
    result_text = reader.readtext(result)

    # 初始化参数
    sign_left_start = None
    sign_top_start = None
    sign_top_end = None
    sign_left_end = None
    last_15_bottom_end = None

    # 搜索签名和最后一个15
    for item in reversed(result_text):
        box, text, _ = item
        if "签名" in text:
            top_left, top_right, bottom_right, bottom_left = box
            x1, y1 = top_left
            x2, y2 = bottom_right
            if sign_left_start is None:
                sign_left_start = x1
                sign_top_start = y1
                sign_top_end = y2
                sign_left_end = x2
            else:
                sign_left_start = min(sign_left_start, x1)
                sign_top_start = min(sign_top_start, y1)
                sign_top_end = max(sign_top_end, y2)
                sign_left_end = max(sign_left_end, x2)
        elif "15" in text and last_15_bottom_end is None:
            _, _, _, bottom_left = box
            _, y = bottom_left
            last_15_bottom_end = y

    # 计算高度
    text_high = sign_top_end - sign_top_start
    total_high = last_15_bottom_end - sign_top_start
    field_high = (total_high - text_high) / 15
    blank_high = (field_high - text_high) / 2

    # 计算宽度
    box_width = 3.5 * (sign_left_end - sign_left_start)

    # 根据计算的高度和宽度，生成签名框
    sign_boxes = []
    for i in range(15):
        top = sign_top_end + (2 * blank_high) + i * field_high
        bottom = top + field_high - blank_high * 2
        left = sign_left_start + blank_high
        right = left + box_width
        box = [[left, top], [right, top], [right, bottom], [left, bottom]]
        # 画出红色的框
        cv2.polylines(result, [np.array(box, np.int32).reshape((-1, 1, 2))], True, (0, 0, 255), 2)
        sign_boxes.append(box)  # 将这个框的位置添加到sign_boxes列表中

    # 转换为灰度图像
    corrected_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, corrected_binary = cv2.threshold(corrected_gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 使用Hough变换检测直线
    # cv2.HoughLinesP()函数用于检测图像中的直线，这个函数的参数设置会影响检测结果。
    #
    # rho（本例中值为1）：距离分辨率，以像素为单位。这是创建累加器来检测线段的分辨率。
    # theta（本例中值为np.pi / 180）：角度分辨率，以弧度为单位。这是创建累加器来检测线段的分辨率。
    # threshold（本例中值为100）：只有累加器中的值高于阈值的线段才被返回。阈值参数越大，可以检测到的直线越少。
    # minLineLength（本例中值为50）：最小线段长度。线段长度小于此值的线段将不被检测。
    # maxLineGap（本例中值为20）：线段之间允许的最大间隙，如果间隙小于此值，这两条线段将被视为一条线段。
    lines = cv2.HoughLinesP(corrected_binary, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=50)

    # 移除检测到的直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(corrected_binary, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # # todo test
    # cv2.imshow('corrected_binary', corrected_binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 初始化签名数量
    sign_count = 0

    # 创建蓝色遮罩
    blue_mask = np.zeros_like(result)
    blue_mask[:, :] = (250, 206, 135)  # BGR for blue 135 206 250

    # 对每个签名框进行处理
    for i, box in enumerate(sign_boxes):
        top_left, top_right, bottom_right, bottom_left = box
        x = int(top_left[0])
        y = int(top_left[1])
        w = int(top_right[0] - top_left[0])
        h = int(bottom_right[1] - top_right[1])

        # 在原图上画出签名框
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 4)

        # 提取签名框
        signature_box = corrected_binary[y:y + h, x:x + w]

        # 计算黑色像素的数量
        count = np.sum(signature_box == 255)

        # 如果黑色像素数量超过一个阈值，那么我们可以认为这个框内有签名
        if count > 700 * re_size * re_size:  # 这只是一个例子，你需要根据你的图像来调整这个阈值
            print('签名框', i + 1, f'找到签名,笔记数为{count}')
            sign_count += 1

            # # todo test
            # cv2.imshow(f'{i+1}_signature_box_{count}', signature_box)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 增加透明的蓝色覆盖
            alpha = 0.5  # transparency level
            added_image = cv2.addWeighted(result[y:y + h, x:x + w], alpha, blue_mask[y:y + h, x:x + w], 1 - alpha, 0)
            result[y:y + h, x:x + w] = added_image
    return result, sign_count


# 创建标题和说明
st.set_page_config(
    page_title="签到表识别Demo",
    layout="wide",  # 将这里改为"wide"
)

# 创建标题和说明
st.title('签名识别Demo')

col1, col2 = st.columns([1, 2])

with col1:
    # 添加模板下载链接
    with open("static/template.xlsx", "rb") as file:
        file_data = file.read()

    st.download_button(
        label="下载模板",
        data=file_data,
        file_name='template.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

    # 上传图片
    uploaded_file = st.file_uploader('选择一张图片', type=['jpg', 'jpeg', 'png'])

with col2:
    # 处理按钮
    if st.button('开始处理'):
        if uploaded_file is not None:
            with st.spinner('签名识别中...'):
                # 将文件对象转换为字节数组
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

                # 使用 OpenCV 读取图片
                cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                cv_image = cv2.resize(cv_image, (0, 0), fx=re_size, fy=re_size)

                # 在此处添加您的图像处理代码
                # 返回处理后的图像和文本结果
                processed_image, text_result = sign_analysis(cv_image)

                # 显示文本结果
                st.write('找到的签名数量：', text_result, '个。')

                # 将 OpenCV 图像转换为 RGB 颜色空间
                rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

                # 将 OpenCV 图像转换为 PIL 图像
                pil_image = Image.fromarray(rgb_image)

                # 显示处理后的图像
                st.image(pil_image, caption='处理后的图像')
        else:
            st.write('请先上传一张图片')
