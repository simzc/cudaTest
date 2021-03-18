#include<iostream>
using namespace std;

void chageVal(int* a)
{
	*a = 100;
	*(a + 1) = 150;
	cout << a << endl;
	cout << a + 1 << endl;
}

int main()
{
	int size(4);
	int* a = new int[size];

	for (int i = 0; i < size; i++)
	{
		a[i] = i;
	}

	int* ptr = &a[2];
	chageVal(ptr);

	for (int i = 0; i < size; i++)
	{
		cout << a[i] << endl;
		cout << &a[i] << endl;

	}

	delete[]a;

	return 0;
}