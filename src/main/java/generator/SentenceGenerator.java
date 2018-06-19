/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package generator;

import commonclasses.TaggedSentence;
import java.lang.reflect.InvocationTargetException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author I353540
 */
public class SentenceGenerator implements Generator<TaggedSentence> {

    @Override
    public TaggedSentence next() {
        try {
            return TaggedSentence.class.getConstructor().newInstance();
        } catch (SecurityException | IllegalArgumentException | NoSuchMethodException | InstantiationException | IllegalAccessException | InvocationTargetException ex) {
            Logger.getLogger(SentenceGenerator.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }

    public TaggedSentence next(int id,String words) {
        try {
            return TaggedSentence.class.getConstructor().newInstance();
        } catch (SecurityException | IllegalArgumentException | InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException ex) {
            Logger.getLogger(SentenceGenerator.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }

}
