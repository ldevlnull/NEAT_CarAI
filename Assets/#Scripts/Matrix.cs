using System;
using Newtonsoft.Json;
using static UnityEngine.Random;

[Serializable]
public class Matrix
{
    private const double MatrixMutationCoefficient = 7;
    
    public int ColsCount { get; set;  }
    public int RowsCount { get; set; }

    public double[][] DataMatrix { get; set; }

    [JsonConstructor]
    public Matrix(int rows, int cols, double[][] dataMatrix)
    {
        RowsCount = rows;
        ColsCount = cols;
        DataMatrix = dataMatrix;
    }
    public Matrix(int rows, int cols)
    {
        ColsCount = cols;
        RowsCount = rows;
        DataMatrix = new double[rows][];
        for (var rowI = 0; rowI < RowsCount; rowI++)
            DataMatrix[rowI] = new double[cols];
    }

    public double this[int rowI, int colJ]
    {
        get
        {
            if ((uint) rowI >= (uint) RowsCount || (uint) colJ >= (uint) ColsCount)
                throw new ArgumentOutOfRangeException();
            return DataMatrix[rowI][colJ];
        }
        set
        {
            if ((uint) rowI >= (uint) RowsCount || (uint) colJ >= (uint) ColsCount)
                throw new ArgumentOutOfRangeException();
            DataMatrix[rowI][colJ] = value;
        }
    }

    public static Matrix operator *(Matrix a, Matrix b)
    {
        if (a.ColsCount != b.RowsCount)
            throw new ArgumentException(
                "Wrong matrix's sizes! The columns count of matrix A must be equal to rows count of matrix B.");

        var targetColsCount = b.ColsCount;
        var targetRowsCount = a.RowsCount;

        var target = new Matrix(targetRowsCount, targetColsCount);

        for (var rowI = 0; rowI < targetRowsCount; rowI++)
        {
            for (var colJ = 0; colJ < targetColsCount; colJ++)
            {
                var cell = 0d;
                for (var k = 0; k < a.ColsCount; k++)
                    cell += a[rowI, k] * b[k, colJ];
                target[rowI, colJ] = cell;
            }
        }

        return target;
    }

    public static Matrix operator +(Matrix a, double b)
    {
        var r = new Matrix(a.RowsCount, a.ColsCount);
        for (var rowI = 0; rowI < r.RowsCount; rowI++)
        for (var colJ = 0; colJ < r.ColsCount; colJ++)
            r[rowI, colJ] = a[rowI, colJ] + b;
        return r;
    }

    public Matrix PointwiseTanh()
    {
        var r = new Matrix(RowsCount, ColsCount);
        for (var rowI = 0; rowI < r.RowsCount; rowI++)
        for (var colJ = 0; colJ < r.ColsCount; colJ++)
            r[rowI, colJ] = (float) Math.Tanh(this[rowI, colJ]);
        return r;
    }
    
    public Matrix Mutate()
    {
        var randomPoints = Range(1, (float)((RowsCount * ColsCount) / MatrixMutationCoefficient));
        var newMatrix = this;
        for (var i = 0; i < randomPoints; i++)
        {
            var randomRowIndex = Range(0, newMatrix.RowsCount);
            var randomColumnIndex = Range(0, newMatrix.ColsCount);
            var val = newMatrix[randomRowIndex, randomColumnIndex] + Range(-1f, 1f);
            newMatrix[randomRowIndex, randomColumnIndex] = (val > 1f) ? 1f : (val < -1f) ? -1f : val;
        }

        return newMatrix;
    }

    public override string ToString()
    {
        var str = "";
        for (var rowI = 0; rowI < RowsCount; rowI++)
        {
            for (var colJ = 0; colJ < ColsCount; colJ++)
            {
                if (colJ != 0)
                    str += ", ";
                str += DataMatrix[rowI][colJ];
            }

            str += "\n\r";
        }

        return str;
    }

    public double[] ToOneDimensionalArray()
    {
        if (DataMatrix.Length != 1)
            throw new ArgumentException("This array is not one-dimensional! Please use 'ToMultiDimensionalArray()'");
        return DataMatrix[0];
    }

    public double[][] ToMultiDimensionalArray()
    {
        if (DataMatrix.Length == 1)
            throw new ArgumentException("This array is one-dimensional! Please use 'ToOneDimensionalArray()'");
        return DataMatrix;
    }

    public void Clear()
    {
        DataMatrix = new double[RowsCount][];
        for (var rowI = 0; rowI < RowsCount; rowI++)
            DataMatrix[rowI] = new double[ColsCount];
    }
}