<template>
  <div id="app">
    <activeLearning></activeLearning>
  </div>
</template>

<script>
import ActiveLearning from './components/ActiveLearning'

export default {
  name: "app",
  data() {
  },
  methods: {
  },
  components: {
    'activeLearning': ActiveLearning
  }
};
</script>

<style scoped>
h1{
  font-family: "Arial"
}
</style>
