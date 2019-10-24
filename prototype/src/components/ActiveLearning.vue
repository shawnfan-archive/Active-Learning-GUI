<template>
  <div>
    <app-header></app-header>
    <p>Image ID: {{ activationMap.image_id }}</p>
    <p>Image group: {{ activationMap.disease }}</p>
    <p>Click on activation map to correct activations</p>
    <button v-on:click="setDrawMode('draw')">Draw</button>
    <button v-on:click="setMarkerSize(10)">10</button>
    <button v-on:click="setMarkerSize(20)">20</button>
    <button v-on:click="setMarkerSize(30)">30</button>
    <button v-on:click="setDrawMode('erase')">Eraser</button>
    <v-stage ref="stage" :config="configStage" v-on:mousemove="correctActivation">
      <v-layer ref="layer" :config="configLayer" v-on:mousedown="toggleDrawing">
        <v-image :config="{
          image: image
        }"></v-image>
        <v-rect 
          v-for="area in activationMap.activation"
          :key="area.id"
          :config="{x:scale*area.x, y:scale*area.y, width:scale*1, height:scale*1, fill:'white', opacity:0.5}"
        ></v-rect>
        <v-circle
          v-for="marker in markers"
          :key="marker.markerID"
          :config="{x: marker.x, y: marker.y, radius: marker.markerSize, fill: 'blue', id:marker.markerID}"
        ></v-circle>
      </v-layer>
    </v-stage>
    <button v-on:click="onSubmit">Save</button>
    <button v-on:click="retrainModel">Retrain Model</button>
    <app-footer></app-footer>
  </div>
</template>>

<script>
import axios from "axios";
import Header from "./Header";
import Footer from "./Footer";

export default {
  data() {
    return {
      activationMap: {},
      drawMode: "draw",
      drawing: false,
      markerSize: 10,
      markers: [],
      configStage: {
        width: 500,
        height: 500
      },
      configLayer: {},
      scale: 8,
      image: null
    };
  },
  methods: {
    getActivationMap: function() {
      const path = "http://localhost:5000/active_learning";
      axios
        .get(path)
        .then(res => {
          this.activationMap = res.data.activation_maps;
        })
        .catch(error => {
          console.error(error);
        });
    },
    setDrawMode: function(mode) {
      this.drawMode = mode;
    },
    setMarkerSize: function(size) {
      this.drawMode = "draw";
      this.markerSize = size;
    },
    toggleDrawing: function(event) {
      if (this.drawing === false) {
        this.drawing = true;
      } else {
        this.drawing = false;
      }
      console.log(this.drawing);
    },
    correctActivation: function(event) {
      console.log(event.target.className);
      if (this.drawMode === "draw" && this.drawing === true) {
        let pos = this.$refs.stage.getStage().getPointerPosition();
        let position = {
          x: pos.x,
          y: pos.y,
          markerSize: this.markerSize,
          markerID: this.markers.length
        };
        this.markers.push(position);
      } else if (
        this.drawMode === "erase" &&
        event.target.className === "Circle" &&
        this.drawing === true
      ) {
        console.log(event.target.attrs.id);
        this.markers = this.markers.filter(
          marker => marker.markerID !== event.target.attrs.id
        );
      }
    },
    saveData: function(payload) {
      const path = "http://localhost:5000/active_learning";
      axios
        .post(path, payload)
        .then(() => {
          alert("Progress saved!");
          this.getActivationMap();
        })
        .catch(error => {
          console.log(error);
          this.getActivationMap();
        });
    },
    retrainModel: function(event) {
      alert("Training in progress...");
    },
    onSubmit: function(event) {
      const payload = { corrections: this.markers };
      this.saveData(payload);
    }
  },
  components: {
    "app-header": Header,
    "app-footer": Footer
  },
  created() {
    this.getActivationMap();
  },
  mounted() {
    let image = new Image();
    image.src = require("../assets/pic1.jpeg");
    this.image = image;
  },
};
</script>

<style scoped>
</style>