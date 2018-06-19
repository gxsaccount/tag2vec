/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.doc2vec;

import java.nio.DoubleBuffer;
import org.tensorflow.Tensor;

/**
 *
 * @author 447653585@qq.com
 */
public class NewClass {
    public static void main(String[] args) {
        long[] shape = new long[]{669944};
        DoubleBuffer doubleBuffer =DoubleBuffer.allocate(669944);
        for (int i = 0; i < 669944; i++) {
            doubleBuffer.put((double)i);
        }
        doubleBuffer.flip();
        Tensor t = Tensor.create( shape,doubleBuffer);
    }
}
