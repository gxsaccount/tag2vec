/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.doc2vec;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 *
 * @author hadoop
 */
public class TensorflowUtil {

    public static Tensor concat(Tensor t0, Tensor t1, int concatDim) {
        Tensor reslut;
        try (Graph g = new Graph();Session s = new Session(g)) {
            Output x = g.opBuilder("Placeholder", "value1").setAttr("dtype", t0.dataType()).build().output(0);
            Output x1 = g.opBuilder("Placeholder", "value2").setAttr("dtype", t1.dataType()).build().output(0);
            Output y = g.opBuilder("Placeholder", "concat_dim").setAttr("dtype", DataType.INT32).build().output(0);
            Output z = g.opBuilder("Concat", "z").addInput(y).addInputList(new Output<?>[]{x, x1}).build().output(0);
            reslut = s.runner().feed("concat_dim", Tensor.create(concatDim)).feed("value1", t0).feed("value2", t1).fetch("z").run().get(0);
        }
        return reslut;
    }
}
