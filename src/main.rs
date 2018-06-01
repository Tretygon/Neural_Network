#[macro_use]
extern crate serde_derive;
extern crate rand;
extern crate time;
extern crate serde;
extern crate serde_json;
extern crate mnist;

use std::io;
use rand::Rng;
use time::Duration;
use std::result::Result::Err;
use mnist::{Mnist, MnistBuilder};
use std::fs;

const E: f32 = 2.71828_f32;
const LEARN_RATE: f32 = 0.01_f32;
const LEARN_FUNCTION: fn(f32)->f32 = sigmoid;
const PATH: &'static str = "NN.txt" ;

fn main() {
    
    let nn = match NeuralNetwork::load(){
        Ok(nn)=> nn,
        Err(_) => {
            let sizes = vec![28*28,16,16,10];
            NeuralNetwork::new(sizes)
        }
    };
    let Mnist {mut trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .base_path("samples/")
        .training_images_filename("training_images.idx3-ubyte")
        .training_labels_filename("training_labels.idx1-ubyte")
        .test_images_filename("test_images.idx3-ubyte")
        .test_labels_filename("test_labels.idx1-ubyte")
        .training_set_length(50000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    println!("\n\n");
    let data : Vec<f32>= Vec::from(&mut trn_img[0..(28*28)]).into_iter().map(|a| a as f32).collect();
    for r in nn.run(data){
        println!("{}", r);
    }
    nn.save();
}

enum LearnDuration{
    Time(Duration),
    Epochs(u64),
}

#[derive(Serialize, Deserialize)]
struct NeuralNetwork{
    weights: Vec<Vec<Vec<f32>>>,
    sizes: Vec<usize>
}
impl NeuralNetwork{
    fn new (sizes: Vec<usize>, )-> NeuralNetwork{
        
        assert![sizes.len() > 1, "network cannot be empty"];
        assert![(&sizes).iter().all(|a|a > &0_usize), "no layer can be empty"];

        let mut rng = rand::thread_rng();
        let mut weights:Vec<Vec<Vec<f32>>> = vec![];
        for layer_num in 0..(sizes.len() - 1){
            let mut layer : Vec<Vec<f32>>= vec![];
            for y in 0..sizes[layer_num + 1]{
                let mut neuron: Vec<f32> = vec![];
                for x in 0..sizes[layer_num]{
                    neuron.push(rng.gen_range(-1_f32, 1_f32));
                }
                layer.push(neuron);
            }
            weights.push(layer);
        }
        
        NeuralNetwork{
            weights, 
            sizes
        }
    }
    fn load()->Result<NeuralNetwork, io::Error>{
        let data = fs::read_to_string(PATH)?;
        let deserialized: NeuralNetwork = serde_json::from_str(&data)?;
        Ok(deserialized)
    }
    fn save(&self){
        let serialized = serde_json::to_string(&self).unwrap();
        fs::write(PATH, serialized).expect(&format!["unable to write to file: {}", PATH]);
    }
    fn run(&self, mut data: Vec<f32>) -> Vec<f32>{
        for layer in &self.weights{
            data = process_layer(&data, layer, LEARN_FUNCTION);
        }
        data
    }

}
fn process_layer(neurons:&Vec<f32>, weights: &Vec<Vec<f32>>, f: fn(f32)->f32) -> Vec<f32>{
    let mut result = vec![];
    for (neuron, row) in neurons.iter().zip(weights.iter()){
        let mut sum = 0_f32;
        for weight in row{
            sum += neuron * weight;
        }
        result.push(f(sum));
    }
    result
}
pub fn sigmoid(x: f32)->f32{
    1_f32/(E.powf(-x) +  1_f32)
}
pub fn relu(x: f32)-> f32{
    x > 0_f32 ? x 0_f32
}
pub fn tanh(x: f32)-> f32{
    let sqr = E.powf(2_f32*x);
    (sqr - 1_f32) / (sqr + 1_f32)
}