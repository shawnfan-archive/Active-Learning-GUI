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
    <v-stage ref="stage" :config="configStage" v-on:mousedown="correctActivation">
      <v-layer ref="layer" :config="configLayer">
        <v-image :config="configImage"></v-image>
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
      markerSize: 10,
      markers: [],
      configStage: {
        width: 500,
        height: 500,
      },
      configLayer: {
      },
      configImage: {
        image: new Image()
      }
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
    correctActivation: function(event) {
      if(this.drawMode === "draw") {
        let pos = this.$refs.stage.getStage().getPointerPosition();
        let position = { x: pos.x, y: pos.y, markerSize: this.markerSize, markerID: this.markers.length };
        this.markers.push(position);
      } else if(event.target.className === "Circle") {
        console.log(event.target.attrs.id)
        this.markers = this.markers.filter(marker => marker.markerID !== event.target.attrs.id)
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
      const payload = {corrections: this.markers};
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
    this.configImage.image.src =
      "https://www.radiologyinfo.org/gallery-items/images/picbrain.jpg";
  }
};
</script>

<style scoped>
</style>