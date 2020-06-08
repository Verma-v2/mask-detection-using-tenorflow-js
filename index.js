const webcamElement= document.getElementById('webcam');
webcamElement.autoplay = true;

var mask1;
let net;
let mask;

let isPredicting = true;
app();


async function app(){
	console.log('Loading model..');
	net= await tf.automl.loadImageClassification('model.json');
	console.log('Successfully loaded model');
	
	const webcam = await tf.data.webcam(webcamElement);
	while(isPredicting){
	const img = await webcam.capture();
	const result = await net.classify(img);
	
	console.log(result);
	
	document.getElementById("predictions-mask").innerText=result['0']['label']+": "+Math.round(result['0']['prob']*100)+"%";
	document.getElementById("predictions-no-mask").innerText= Math.round(result['1']['prob']*100);
	


	
	img.dispose();
	
	await tf.nextFrame();
	
	}
	
	
	
}






