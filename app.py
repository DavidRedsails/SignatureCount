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

    # 初始化左边距和列的宽度
    column_left = None
    column_width = result.shape[1]  # 列宽度设为图片的宽度
    row_height = result.shape[0] / 16

    # 搜索签名
    sign_row = None
    for item in result_text:
        box, text, _ = item
        if "签名" in text:
            top_left, top_right, bottom_right, bottom_left = box
            x, y = top_left
            column_left = x  # 记录签名的左边距
            sign_row = int(y // row_height)  # 确定签名所在的行
            # 用绿色框表示出签名
            cv2.polylines(result, [np.array(box, np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            break  # 找到签名后就跳出循环

    # 根据签名的左边距和图片的高度，均匀分成16个部分，并用红色的框标记出来
    sign_boxes = []
    if column_left is not None:
        for i in range(16):
            if i <= sign_row:  # 跳过在"签名"行及其以上的行
                continue
            top_left = [column_left + 10, i * row_height + 10]
            bottom_right = [column_width - 10, (i + 1) * row_height - 10]
            box = [top_left, [bottom_right[0], top_left[1]], bottom_right, [top_left[0], bottom_right[1]]]
            # 用红色框表示出每个部分
            cv2.polylines(result, [np.array(box, np.int32).reshape((-1, 1, 2))], True, (0, 0, 255), 2)
            sign_boxes.append(box)  # 将这个框的位置添加到sign_boxes列表中

    # 转换为灰度图像
    corrected_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, corrected_binary = cv2.threshold(corrected_gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 使用Hough变换检测直线
    lines = cv2.HoughLinesP(corrected_binary, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # 移除检测到的直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(corrected_binary, (x1, y1), (x2, y2), (0, 0, 0), 2)

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
        if count > 800 * re_size * re_size:  # 这只是一个例子，你需要根据你的图像来调整这个阈值
            print('Box', i + 1, 'has a signature.')
            sign_count += 1

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
