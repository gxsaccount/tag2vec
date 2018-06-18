/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package commonclasses;

import java.util.List;

/**
 *
 * @author I353540
 */
public class TaggedDocument {
    List<String> tags;
    List<String> words;

    public TaggedDocument(List<String> tags, List<String> words) {
        this.tags = tags;
        this.words = words;
    }

    public List<String> getTags() {
        return tags;
    }

    public List<String> getWords() {
        return words;
    }
    
}
