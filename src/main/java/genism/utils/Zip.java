/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package genism.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author I353540
 */
public class Zip {
    public static <T> List<List<T>> zip(List<T>... lists) {
    List<List<T>> zipped = new ArrayList<List<T>>();
    for (List<T> list : lists) {
        for (int i = 0, listSize = list.size(); i < listSize; i++) {
            List<T> list2;
            if (i >= zipped.size())
                zipped.add(list2 = new ArrayList<T>());
            else
                list2 = zipped.get(i);
            list2.add(list.get(i));
        }
    }
    return zipped;
}

public static void main(String[] args) {
        List<Integer> x = Arrays.asList(1, 2, 3);
        List<Integer> y = Arrays.asList(4, 5, 6);
        List<List<Integer>> zipped = zip(x, y);
        System.out.println(zipped);
}
}
