// 为简单期间，本代码没有实现图像的resize功能，而是直接把摄像头的输出分辨率设置为56*56
// 如果是分辨率较大的图片，要先resize到56*56再出入网络，如果直接使用大分辨率图像进行预测，结果会很差
// 另外单片机有限的RAM也没法使用大分辨率的图像
// 如果有不理解的可以参考其他几个文件夹里的python文件或者ncnn文件夹里的cpp文件


// 定义图像数组
// 我使用的OV7725返回的是16bit的RGB565
uint16_t image_data[56*56] = {0};

void GetImage(uint16_t * image)
{
	uint16_t color;
	for(int i=0; i<Image.height; i++)
	{
		for(int j=0; j<Image.width; j++)
		{
			// 获取像素值
            // 我是按照先行后列的顺序存储的
			image[j+i*Image.width]=color;
		}
	}
}

// 这个函数展示了如何从16bit RGB565中提取出R G B
uint32_t ReadColor(uint16_t x, uint16_t y)
{
	uint16_t color = image_data[x+y*Image.width];
	uint8_t r,g,b;
	r = (color&0xF800)>>9;
	g = (color&0x07E0)>>3;
	b = (color&0x001F)<<3;
	return (r<<16)|(g<<8)|b;
}

// 定义网络输入数组
AI_ALIGNED(32)
static ai_i8 in_data[AI_NETWORK_IN_1_SIZE];

// 定义网络输出数组
AI_ALIGNED(32)
static ai_i8 out_data[AI_NETWORK_OUT_1_SIZE];

// 人脸方框的左上右下像素坐标
int x1, y1, x2, y2;
// yoloface的anchor尺寸
uint8_t anchors[3][2] = {{9, 14}, {12, 17}, {22, 21}};

// 网络预处理函数，任务是把图像传入网络
// 网络量化之后会给出对应的输入和输出的缩放比例和偏移量，可以使用netron查看yoloface_int8.tflite得到
// 网络训练的时候需要把图像归一化，即从0~255缩放到0~1，然后再根据量化得到的缩放比例和偏移量计算
// 最后量化的网络输入图像范围是-128~127，正好是RGB分别减去128即可
void prepare_yolo_data()
{
	for(int i = 0; i < 56; i++)
	{
		for(int j = 0; j < 56; j++)
		{
			uint16_t color = image_data[j+i*Image.width];
			// 这里要注意，网络的输入张量维度是BHWC，对应1*56*56*3，通道顺序是RGB
            // 所以输入数组的存储顺序应该是先行后列，颜色是R G B
			in_data[(j+i*56)*3] = (int8_t)((color&0xF800)>>9) - 128;
			in_data[(j+i*56)*3+1] = (int8_t)((color&0x07E0)>>3) - 128;
			in_data[(j+i*56)*3+2] = (int8_t)((color&0x001F)<<3) - 128;
		}
	}
}

// 定义sigmoid函数
float sigmoid(float x)
{
	float y = 1/(1+expf(x));
	return y;
}

// 网络后处理函数
// 注意，正常的YOLO后处理都应该包含非极大值抑制NMS操作，但因为我比较懒所以没有加，这里只是根据置信度做了初步提取
void post_process()
{
	int grid_x, grid_y;
	float x, y, w ,h;
	for(int i = 0; i < 49; i++)
	{
		for(int j = 0; j < 3; j++)
		{
            // 网络输出维度是1*7*7*18
            // 其中18维度包含了每个像素预测的三个锚框，每个锚框对应6个维度，依次为x y w h conf class
            // 当然因为这个网络是单类检测，所以class这一维度没有用
            // 如果对YOLO不熟悉的话，建议去学习一下yolov3
			int8_t conf = out_data[i*18+j*6+4];
            // 这里的-9是根据网络量化的缩放偏移量计算的，对应的是70%的置信度
            // sigmoid((conf+15)*0.14218327403068542) < 0.7 ==> conf > -9
			if(conf > -9)
			{
				grid_x = i % 7;
				grid_y = (i - grid_x)/7;
				// 这里的15和0.14218327403068542就是网络量化后给出的缩放偏移量
				x = ((float)out_data[i*18+j*6+1]+15)*0.14218327403068542f;
				y = ((float)out_data[i*18+j*6]+15)*0.14218327403068542f;
				w = ((float)out_data[i*18+j*6+3]+15)*0.14218327403068542f;
				h = ((float)out_data[i*18+j*6+2]+15)*0.14218327403068542f;
                // 网络下采样三次，缩小了8倍，这里给还原回56*56的尺度
				x = (sigmoid(x)+grid_x) * 8;
				y = (sigmoid(y)+grid_y) * 8;
				w = expf(w) * anchors[j][0];
				h = expf(h) * anchors[j][1];
				y2 = (x - w/2);
				y1 = (x + w/2);
				x1 = y - h/2;
				x2 = y + h/2;
				if(x1 < 0) x1 = 0;
				if(y1 < 0) y1 = 0;
				if(x2 > 55) x2 = 55;
				if(y2 > 55) y2 = 55;
                // 绘制方框，左上角坐标为(x1, y1)，左下角坐标为(x2, y2)
                // 注意，如果输入图像是缩放到56*56再输入网络的话，这里的坐标还要乘以图像的缩放系数
				GUI_Rectangle(x1, y1, x2, y2, RED);
			}
		}
	}
}

int main()
{
    // 此处省略了其他代码
    // 调用yoloface.c里面的神经网络初始化函数
    aiInit();

    while(1)
    {
        // 获取图像数据
        GetImage(image_data);
        // 图像预处理
        prepare_yolo_data();
        // 调用yoloface.c里面的网络推理函数
        aiRun(in_data, out_data);
        // 图像后处理
        post_process();
    }
}
