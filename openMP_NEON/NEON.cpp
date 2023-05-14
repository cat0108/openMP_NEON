
//��linux��ARMָ��Ͽ�ִ��
#include<iostream>
#include<stdio.h>
#include<arm_neon.h>
#include<sys/time.h>
#include<omp.h>
using namespace std;
float gdata[10000][10000];
float gdata2[10000][10000];
float gdata1[10000][10000];
float gdata3[10000][10000];
int Num_thread = 8;
void Initialize(int N)
{
	for (int i = 0; i < N; i++)
	{
		//���Ƚ�ȫ��Ԫ����Ϊ0���Խ���Ԫ����Ϊ1
		for (int j = 0; j < N; j++)
		{
			gdata[i][j] = 0;
			gdata1[i][j] = 0;
			gdata2[i][j] = 0;
			gdata3[i][j] = 0;
		}
		gdata[i][i] = 1.0;
		//�������ǵ�λ�ó�ʼ��Ϊ�����
		for (int j = i + 1; j < N; j++)
		{
			gdata[i][j] = rand();
			gdata1[i][j] = gdata[i][j] = gdata2[i][j] = gdata3[i][j];
		}
	}
	for (int k = 0; k < N; k++)
	{
		for (int i = k + 1; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				gdata[i][j] += gdata[k][j];
				gdata1[i][j] += gdata1[k][j];
				gdata2[i][j] += gdata2[k][j];
				gdata3[i][j] += gdata3[k][j];
			}
		}
	}

}

void Normal_alg(int N)
{
	int i, j, k;
	for (k = 0; k < N; k++)
	{
		for (j = k + 1; j < N; j++)
		{
			gdata1[k][j] = gdata1[k][j] / gdata1[k][k];
		}
		gdata1[k][k] = 1.0;
		for (i = k + 1; i < N; i++)
		{
			for (j = k + 1; j < N; j++)
			{
				gdata1[i][j] = gdata1[i][j] - (gdata1[i][k] * gdata1[k][j]);
			}
			gdata1[i][k] = 0;
		}
	}
}
//ֻ�Ե�һ��ѭ���Ż�
void omp_NEON(int n)
{
#pragma omp parallel num_threads(Num_thread)
	{
		int i, j, k;
		float32x4_t r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
		for (k = 0; k < n; k++)
		{
#pragma omp single
			{
				r0 = vmovq_n_f32(gdata3[k][k]);//��ʼ��4������gdatakk������
				for (j = k + 1; j + 4 <= n; j += 4)
				{
					r1 = vld1q_f32(gdata3[k] + j);
					r1 = vdivq_f32(r1, r0);//������������λ���
					vst1q_f32(gdata3[k], r1);//���������·Ż��ڴ�
				}
				//��ʣ�಻��4�������ݽ�����Ԫ
				for (j; j < n; j++)
				{
					gdata3[k][j] = gdata3[k][j] / gdata3[k][k];
				}
				gdata3[k][k] = 1.0;
			}
			//���϶�Ӧ������һ������ѭ���Ż���SIMD
#pragma omp for
			for (i = k + 1; i < n; i++)
			{

				r0 = vmovq_n_f32(gdata3[i][k]);
				for (j = k + 1; j + 4 <= n; j += 4)
				{
					r1 = vld1q_f32(gdata3[k] + j);
					r2 = vld1q_f32(gdata3[i] + j);
					r3 = vmulq_f32(r0, r1);
					r2 = vsubq_f32(r2, r3);
					vst1q_f32(gdata3[i] + j, r2);
				}
				for (j; j < n; j++)
				{
					gdata3[i][j] = gdata3[i][j] - (gdata3[i][k] * gdata3[k][j]);
				}
				gdata3[i][k] = 0;
			}
		}
	}
}


//��ȫ�������Ż�
void nomal_NEON(int n)
{
	int i, j, k;
	float32x4_t r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
	for (k = 0; k < n; k++)
	{
		r0 = vmovq_n_f32(gdata[k][k]);//�ڴ治�������ʽ���ص������Ĵ�����
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			r1 = vld1q_f32(gdata[k] + j);
			r1 = vdivq_f32(r1, r0);//������������λ���
			vst1q_f32(gdata[k], r1);//���������·Ż��ڴ�
		}
		//��ʣ�಻��4�������ݽ�����Ԫ
		for (j; j < n; j++)
		{
			gdata[k][j] = gdata[k][j] / gdata[k][k];
		}
		gdata[k][k] = 1.0;
		//���϶�Ӧ������һ������ѭ���Ż���SIMD

		for (i = k + 1; i < n; i++)
		{

			r0 = vmovq_n_f32(gdata[i][k]);
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = vld1q_f32(gdata[k] + j);
				r2 = vld1q_f32(gdata[i] + j);
				r3 = vmulq_f32(r0, r1);
				r2 = vsubq_f32(r2, r3);
				vst1q_f32(gdata[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata[i][j] = gdata[i][j] - (gdata[i][k] * gdata[k][j]);
			}
			gdata[i][k] = 0;
		}
	}
}

int main()
{
	struct timeval begin, end;
	int n = 1000;
	long long res;
	gettimeofday(&begin, NULL);
	Initialize(n);
	gettimeofday(&end, NULL);
	res = (1000 * 1000 * end.tv_sec + end.tv_usec) - (1000 * 1000 * begin.tv_sec + begin.tv_usec);
	cout << "initalize time:" << res << " us" << endl;

	gettimeofday(&begin, NULL);
	Normal_alg(n);
	gettimeofday(&end, NULL);
	res = (1000 * 1000 * end.tv_sec + end.tv_usec) - (1000 * 1000 * begin.tv_sec + begin.tv_usec);
	cout << "Normal time:" << res << " us" << endl;

	gettimeofday(&begin, NULL);
	omp_NEON(n);
	gettimeofday(&end, NULL);
	res = (1000 * 1000 * end.tv_sec + end.tv_usec) - (1000 * 1000 * begin.tv_sec + begin.tv_usec);
	cout << "omp time:" << res << " us" << endl;

	gettimeofday(&begin, NULL);
	nomal_NEON(n);
	gettimeofday(&end, NULL);
	res = (1000 * 1000 * end.tv_sec + end.tv_usec) - (1000 * 1000 * begin.tv_sec + begin.tv_usec);
	cout << "SIMD time:" << res << " us" << endl;

}

